# src/process/thermal/cooling.py
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import logging
from .thermal_utils import ThermalProperties
from .losses import ThermalCalculator
from src.config import Config

# Configuration du logging
logger = logging.getLogger(__name__)

class CoolingCalculator:
    """
    Calculs de refroidissement avec modèle multi-zones
    Prend en compte :
    - Refroidissement naturel
    - Recirculation forcée
    - Stratification thermique
    - Pertes vers l'environnement
    """
    
    def __init__(self, tank):
        """
        Args:
            tank: instance de Bioreactor ou ProcessTank
        """
        self.tank = tank
        self.thermal_calc = ThermalCalculator(tank)

    def calculate_natural_cooling_rate(
        self,
        temp_fluid: float,    # °C
        temp_ambient: float,  # °C
        mass: float,          # kg
        pressure: float = 1.0,  # bar
        flow_rate: Optional[float] = None,  # m³/h
        detailed: bool = False
    ) -> Dict[str, float]:
        """
        Calcule le taux de refroidissement naturel
        Returns:
            dict avec taux et coefficients
        """
        # Validation températures
        Config.validate_operating_conditions(temp_fluid, pressure)
        Config.validate_operating_conditions(temp_ambient, pressure)
        
        if temp_fluid <= temp_ambient:
            if detailed:
                return {
                    'rate': 0,
                    'power': 0,
                    'properties': ThermalProperties.get_water_properties(
                        temp_fluid, pressure
                    )
                }
            return {'rate': 0, 'power': 0}
        
        # Calcul pertes avec détails
        losses = self.thermal_calc.calculate_heat_loss(
            temp_fluid, temp_ambient, pressure, flow_rate,
            detailed=True
        )
        q_loss = losses['total_loss']
        
        # Propriétés fluide à température moyenne
        temp_mean = (temp_fluid + temp_ambient) / 2
        props = ThermalProperties.get_water_properties(temp_mean, pressure)
        
        # Taux refroidissement (K/s)
        cooling_rate = q_loss / (mass * props['cp'])
        
        # Protection contre valeurs négatives
        cooling_rate = max(0, cooling_rate)
        
        results = {
            'rate': cooling_rate,
            'power': q_loss,
            'properties': props
        }
        
        if detailed:
            results.update({
                'losses': losses,
                'temp_mean': temp_mean
            })
        
        return results

    def calculate_temperature_distribution(
        self,
        temp_mean: float,    # °C
        delta_t: float,      # K
        height: float,       # m
        pressure: float = 1.0,  # bar
        flow_rate: Optional[float] = None  # m³/h
    ) -> Dict[str, float]:
        """
        Calcule la distribution verticale de température
        avec prise en compte de la stratification
        """
        # Validation
        if height <= 0:
            raise ValueError(f"Hauteur invalide: {height}m")
            
        if abs(delta_t) > Config.ProcessLimits.TEMP_MAX:
            logger.warning(f"Delta T important: {delta_t}°C")
        
        # Calcul Richardson si écoulement
        if flow_rate and flow_rate > 0:
            area = np.pi * (self.tank.geometry.diameter/2000)**2  # m²
            velocity = (flow_rate / 3600) / area
            
            # Reynolds pour stratification
            re = ThermalProperties.calculate_reynolds(
                velocity, height, temp_mean, pressure,
                context="stratification"
            )
            
            # Correction stratification selon turbulence
            if re > Config.CorrelationLimits.RE_MIN_TURB:
                strat_factor = Config.CorrelationLimits.MIN_STRAT_FACTOR  # Mélange fort
            elif re > Config.CorrelationLimits.RE_MAX_LAMINAR:
                # Interpolation zone transition
                factor = (re - Config.CorrelationLimits.RE_MAX_LAMINAR) / \
                        (Config.CorrelationLimits.RE_MIN_TURB - 
                         Config.CorrelationLimits.RE_MAX_LAMINAR)
                strat_factor = 1 - (1 - Config.CorrelationLimits.MIN_STRAT_FACTOR) * \
                              factor
            else:
                strat_factor = 1.0  # Stratification complète
        else:
            velocity = 0
            strat_factor = 1.0
            
        # Gradient thermique avec facteur
        grad_max = Config.CorrelationLimits.MAX_TEMP_GRADIENT * strat_factor
        
        # Points de calcul
        points = 20
        heights = np.linspace(0, height, points)
        
        # Distribution températures
        if strat_factor > 0.9:  # Forte stratification
            # Profil avec tangente hyperbolique
            z_norm = heights / height
            temps = temp_mean + delta_t * \
                   (2 * z_norm - 1) * np.tanh(2*z_norm)
        else:
            # Profil linéaire atténué
            temps = temp_mean + grad_max * (2 * heights/height - 1) * delta_t
            
        return {
            'temp_bottom': temps[0],
            'temp_middle': temps[points//2],
            'temp_top': temps[-1],
            'stratification_factor': strat_factor,
            'gradient': grad_max,
            'profile': list(zip(heights, temps)),
            'reynolds': re if flow_rate else None
        }
        
    def calculate_cooling_time(
        self,
        temp_initial: float,   # °C
        temp_target: float,    # °C
        temp_ambient: float,   # °C
        mass: float,           # kg
        pressure: float = 1.0,  # bar
        flow_rate: Optional[float] = None,  # m³/h
        time_step: Optional[float] = None,  # s
        detailed: bool = False
    ) -> Dict[str, float]:
        """
        Calcule le temps de refroidissement avec stratification
        Returns:
            dict avec temps et paramètres
        """
        # Validation initiale
        Config.validate_operating_conditions(temp_initial, pressure)
        Config.validate_operating_conditions(temp_target, pressure)
        Config.validate_operating_conditions(temp_ambient, pressure)
        
        if temp_target >= temp_initial:
            raise ValueError("Température cible doit être inférieure à initiale")
            
        if temp_target <= temp_ambient:
            logger.warning(
                f"Température cible ({temp_target}°C) inférieure à "
                f"ambiante ({temp_ambient}°C)"
            )
        
        # Pas de temps par défaut
        if time_step is None:
            time_step = Config.NumericalParams.DT_COOLING
        
        time = 0
        temp = temp_initial
        cooling_curve = [(0, temp)]
        evolution_data = []
        
        while temp > temp_target:
            # Calcul refroidissement instantané
            cooling = self.calculate_natural_cooling_rate(
                temp, temp_ambient, mass, pressure, flow_rate,
                detailed=True
            )
            
            # Distribution températures
            height = self.tank.geometry.height_total / 1000
            temp_dist = self.calculate_temperature_distribution(
                temp, cooling['rate'] * time_step, height, pressure, flow_rate
            )
            
            # Évolution température moyenne
            temp -= cooling['rate'] * time_step
            time += time_step
            
            cooling_curve.append((time/3600, temp))
            
            if detailed:
                evolution_data.append({
                    'time': time,
                    'temperature': temp,
                    'cooling': cooling,
                    'distribution': temp_dist
                })
                
            # Protection boucle infinie
            if time > Config.NumericalParams.MAX_TIME:
                logger.warning("Temps maximum atteint")
                break
        
        results = {
            'time_hours': time/3600,
            'cooling_curve': cooling_curve,
            'final_temperature': temp,
            'total_energy': mass * cooling['properties']['cp'] * \
                          (temp_initial - temp_target),
            'average_rate': (temp_initial - temp)/time if time > 0 else 0
        }
        
        if detailed:
            results['evolution_data'] = evolution_data
            
        return results
    
    def calculate_temperature_profile(
        self,
        temp_initial: float,   # °C
        temp_ambient: float,   # °C
        duration: float,       # heures
        pressure: float = 1.0,  # bar
        flow_rate: Optional[float] = None,  # m³/h
        time_steps: int = 100,
        detailed: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Calcule l'évolution complète du refroidissement
        avec stratification
        Returns:
            dict avec arrays temps et températures
        """
        # Validation
        Config.validate_operating_conditions(temp_initial, pressure)
        Config.validate_operating_conditions(temp_ambient, pressure)
        
        # Protection time_steps
        time_steps = max(10, min(time_steps, 1000))
        
        # Initialisation arrays
        times = np.linspace(0, duration * 3600, time_steps)  # secondes
        temps_mean = np.zeros(time_steps)
        temps_top = np.zeros(time_steps)
        temps_bottom = np.zeros(time_steps)
        powers = np.zeros(time_steps)
        richardson = np.zeros(time_steps)
        
        temps_mean[0] = temp_initial
        temps_top[0] = temp_initial
        temps_bottom[0] = temp_initial
        
        dt = duration * 3600 / time_steps  # pas de temps en secondes
        mass = self.tank.geometry.volume_useful  # kg
        height = self.tank.geometry.height_total / 1000  # m
        
        evolution_data = []
        
        for i in range(1, time_steps):
            # Taux de refroidissement actuel
            cooling = self.calculate_natural_cooling_rate(
                temps_mean[i-1], temp_ambient, mass, pressure, flow_rate
            )
            
            # Distribution températures
            temp_dist = self.calculate_temperature_distribution(
                temps_mean[i-1], cooling['rate'] * dt, height, pressure, flow_rate
            )
            
            # Évolution températures
            temps_mean[i] = max(
                temps_mean[i-1] - cooling['rate'] * dt,
                temp_ambient
            )
            temps_top[i] = temp_dist['temp_top']
            temps_bottom[i] = temp_dist['temp_bottom']
            powers[i] = cooling['power']
            richardson[i] = temp_dist.get('reynolds', 0)
            
            if detailed:
                evolution_data.append({
                    'time': times[i],
                    'temp_mean': temps_mean[i],
                    'cooling': cooling,
                    'distribution': temp_dist
                })
            
        results = {
            'times': times/3600,  # conversion en heures
            'temps_mean': temps_mean,
            'temps_top': temps_top,
            'temps_bottom': temps_bottom,
            'powers': powers,
            'richardson': richardson
        }
        
        if detailed:
            results['evolution_data'] = evolution_data
            
        return results