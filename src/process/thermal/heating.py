# src/process/thermal/heating.py
from typing import Dict, Tuple, Optional, Any, Union
import numpy as np
import logging
from .thermal_utils import ThermalProperties
from .losses import ThermalCalculator
from src.process.thermal.thermal_calculations import ThermalCalculations
from src.equipment.heat_system.electrical_heater import ElectricalHeater
from src.process.thermal.thermal_calculations import CirculationHeatTransfer
from src.config import Config

# Configuration du logging
logger = logging.getLogger(__name__)

class HeatingCalculator:
    """
    Calculs de chauffage avec modèle multi-zones et NUT
    Prend en compte :
    - Échangeur à plaques (méthode NUT)
    - Générateur vapeur
    - Stratification dans le tank
    - Pertes thermiques
    """
    
    def __init__(self, tank):
        """
        Args:
            tank: instance de Bioreactor ou ProcessTank
        """
        self.tank = tank
        self.thermal_calc = ThermalCalculator(tank)
        
    def calculate_steam_requirements(
        self,
        power_needed: float,  # kW
        pressure: float,      # bar
        mode: Optional[str] = None,  # 'cip' ou 'mixing'
        detailed: bool = False
    ) -> Dict[str, float]:
        """
        Calcule les besoins en vapeur pour une puissance donnée
        Args:
            mode: mode opératoire pour validation pression
            detailed: si True, retourne détails supplémentaires
        """
        # Validation pression selon mode
        Config.validate_operating_conditions(
            Config.ProcessLimits.TEMP_MAX, 
            pressure, 
            mode
        )
        
        # Propriétés vapeur saturée
        steam_props = ThermalProperties.get_steam_properties(pressure)
        
        # Puissance avec facteur sécurité
        power_design = power_needed * Config.SafetyFactors.THERMAL
        
        # Débit vapeur nécessaire
        steam_flow = generator.calculate_steam_flow(exchanger_perf['power'] / 1000)  # Utilise puissance réelle (W -> kW)
        power_steam = generator.calculate_heating_power(steam_flow) * 1000  # kW -> W
        
        results = {
            'steam_flow': steam_flow,
            'temperature': steam_props['temperature'],
            'power_design': power_design
        }
        
        if detailed:
            results.update({
                'properties': steam_props,
                'power_requested': power_needed
            })
            
        return results
    
    def calculate_exchanger_performance(
        self,
        exchanger,
        flow_rate: float,
        temp_in: float,
        temp_steam: float,
        pressure: float = 1.0,
        fouling: float = 0.9,
        detailed: bool = False
    ) -> Dict[str, float]:
        """
        Calcule les performances de l'échangeur avec méthodes TEMA
        """
        # Validation des paramètres d'entrée
        if np.isnan(temp_in) or np.isnan(temp_steam):
            raise ValueError(f"Températures invalides : temp_in={temp_in}, temp_steam={temp_steam}")
                
        if np.isnan(pressure):
            raise ValueError(f"Pression invalide : {pressure}")

        # Validation températures
        Config.validate_operating_conditions(temp_in, pressure)

        if temp_steam <= temp_in:
            raise ValueError(
                f"Température vapeur ({temp_steam}°C) doit être > "
                f"température entrée ({temp_in}°C)"
            )

        # 1. Calcul condensation vapeur
        h_condensation = ThermalCalculations.calculate_condensation_film(
            temp_steam=temp_steam,
            temp_wall=(temp_steam + temp_in)/2,  # Estimation
            pressure=pressure,
            height=exchanger.specs.height
        )['h']

        # 2. Calcul convection eau
        h_water = exchanger.calculate_plate_heat_transfer(
            flow_rate=flow_rate,
            temp=temp_in,
            pressure=pressure
        )['h']

        # 3. Coefficient global
        k_plate = Config.PlateExchangerLimits.PLATE_MATERIAL_CONDUCTIVITY[exchanger.specs.material]
        U = 1 / (
            1/h_condensation + 
            Config.PlateExchangerLimits.FOULING_STEAM +
            Config.PlateExchangerLimits.PLATE_THICKNESS*1e-3/k_plate +
            1/h_water + 
            Config.PlateExchangerLimits.FOULING_WATER
        ) * Config.PlateExchangerLimits.SAFETY_FACTOR

        # 4. Puissance maximum théorique
        props = ThermalProperties.get_water_properties(temp_in, pressure, context="exchanger")
        mass_flow = flow_rate * props['rho'] / 3600  # kg/s
        q_max = mass_flow * props['cp'] * (temp_steam - temp_in)  # W

        # 5. Puissance disponible limitée par l'échangeur
        power = min(q_max, exchanger.specs.power * 1000)  # W

        results = {
            'power': power,
            'U': U,
            'h_steam': h_condensation,
            'h_water': h_water,
            'q_max': q_max
        }

        if detailed:
            results.update({
                'properties': props,
                'mass_flow': mass_flow,
                'temp_steam': temp_steam,
                'delta_t': temp_steam - temp_in
            })

        return results


    def calculate_temperature_evolution(
        self,
        temp_initial: float,
        temp_ambient: float,
        power_heating: float,
        time_step: float,
        flow_rate: float,
        pressure: float = 1.0
    ) -> Dict[str, float]:
        """Calcule l'évolution de température avec protection des limites"""
        # Protection puissance négative
        if power_heating < 0:
            power_heating = 0
        
        # Protection température initiale
        temp_safe = ThermalProperties.validate_process_temperature(temp_initial)
        
        # Masse d'eau et propriétés
        mass = self.tank.geometry.volume_useful  # kg
        props = ThermalProperties.get_water_properties(temp_safe, pressure)
        
        # Calcul pertes avec température protégée
        losses = self.thermal_calc.calculate_heat_loss(
            temp_safe, temp_ambient, pressure, flow_rate
        )
        q_loss = losses['total_loss']
        
        # Bilan énergétique
        q_net = power_heating - q_loss  # W
        delta_t = (q_net * time_step) / (mass * props['cp'])
        
        # Limitation physique du taux de chauffe
        max_delta_t = time_step * 1.0  # Maximum 1°C/s
        delta_t = np.clip(delta_t, -max_delta_t, max_delta_t)
        
        # Protection température finale
        temp_new = ThermalProperties.validate_process_temperature(temp_safe + delta_t)
        
        return {
            'temp': temp_new,
            'power_net': q_net,
            'power_heat': power_heating,
            'losses': losses,
            'delta_t': delta_t,
            'properties': props
        }
    
    def calculate_heating_profile(
        self,
        exchanger,
        generator,
        temp_initial: float,
        temp_target: float,
        time_step: float = 30,     # s
        max_time: float = 3600*4,  # s 
        flow_rate: float = 4.0,    # m³/h
        pressure: float = 1.0,     # bar
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Calcule le profil complet de montée en température
        """
        # Calcul de n_steps en premier
        n_steps = int(max_time/time_step) + 1

        # Validation initiale plus stricte
        if not isinstance(temp_initial, (int, float)) or np.isnan(temp_initial):
            raise ValueError("Température initiale invalide")
        if not isinstance(temp_target, (int, float)) or np.isnan(temp_target):
            raise ValueError("Température cible invalide")

        # Validation initiale des conditions opératoires
        Config.validate_operating_conditions(temp_initial, pressure)
        Config.validate_operating_conditions(temp_target, pressure)
        
        if temp_target <= temp_initial:
            raise ValueError(
                f"Température cible {temp_target}°C doit être > "
                f"température initiale {temp_initial}°C"
            )
                
        # Initialisation arrays avec protection
        times = np.zeros(n_steps)
        temps = np.zeros(n_steps)
        powers = np.zeros(n_steps)
        losses = np.zeros(n_steps)
        temps[0] = float(temp_initial)
        
        evolution_data = []
        
        # Simulation pas à pas
        for i in range(1, n_steps):
            try:
                # Temps actuel
                times[i] = i * time_step
                
                # 1. Température vapeur
                steam_props = ThermalProperties.get_steam_properties(
                    generator.specs.pressure
                )
                
                steam_temp = steam_props['temperature']
                
                # 2. Performance échangeur (puissance maximale possible)
                exchanger_perf = self.calculate_exchanger_performance(
                    exchanger=exchanger,
                    flow_rate=flow_rate,
                    temp_in=temps[i-1],
                    temp_steam=steam_temp,
                    pressure=pressure
                )

                # 3. Limitation par générateur
                steam_flow = generator.calculate_steam_flow(exchanger_perf['power'] / 1000)  # W -> kW
                power_steam = generator.calculate_heating_power(steam_flow) * 1000  # kW -> W
                
                # 4. Puissance finale disponible
                power = min(exchanger_perf['power'], power_steam)
                powers[i] = power / 1000  # W -> kW
                
                # 5. Évolution température
                evolution = self.calculate_temperature_evolution(
                    temp_initial=temps[i-1],
                    temp_ambient=20,
                    power_heating=power,
                    time_step=time_step,
                    flow_rate=flow_rate,
                    pressure=pressure
                )
                
                temps[i] = evolution['temp']
                losses[i] = evolution['losses']['total_loss'] / 1000  # W -> kW
                
                if detailed:
                    evolution_data.append({
                        'time': times[i],
                        'temperature': temps[i],
                        'steam': steam_props,
                        'exchanger': exchanger_perf,
                        'evolution': evolution
                    })
                
                # Arrêt si température atteinte
                if temps[i] >= temp_target:
                    # Tronquer arrays
                    times = times[:i+1]
                    temps = temps[:i+1]
                    powers = powers[:i+1]
                    losses = losses[:i+1]
                    break
                    
            except Exception as e:
                raise ValueError(f"Erreur à l'étape {i}: {str(e)}")
                    
        # Résultats
        results = {
            'times': times/60,  # conversion en minutes
            'temperatures': temps,
            'powers': powers,
            'losses': losses,
            'duration': times[-1]/60,
            'final_temp': temps[-1],
            'average_power': np.mean(powers),
            'total_energy': np.sum(powers) * time_step/3600  # kWh
        }
        
        if detailed:
            results['evolution_data'] = evolution_data
                
        return results
    
    def calculate_electrical_heating_profile(
        self,
        heater: ElectricalHeater,
        temp_initial: float,   # °C  
        temp_target: float,    # °C
        time_step: float = 30, # s
        voltage: Optional[float] = None,  # V
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Calcule le profil de chauffe avec résistance électrique
        Args:
            heater: Instance ElectricalHeater
            temp_initial: Température initiale fluide
            temp_target: Température cible
            time_step: Pas de temps simulation
            voltage: Tension d'alimentation
            detailed: Si True, retourne détails
        Returns:
            dict avec profil temps/température et paramètres
        """
        try:
            # 1. Validation initiale
            Config.validate_operating_conditions(temp_initial, self.tank.design_pressure)
            Config.validate_operating_conditions(temp_target, self.tank.design_pressure)
            
            if temp_target <= temp_initial:
                raise ValueError(f"Température cible {temp_target}°C <= initiale {temp_initial}°C")

            # 2. Validation tension si spécifiée
            if voltage:
                if not (Config.ElectricalHeaterLimits.MIN_VOLTAGE <= 
                    voltage <= Config.ElectricalHeaterLimits.MAX_VOLTAGE):
                    raise ValueError(f"Tension {voltage}V hors limites")
                    
            # 3. Initialisation arrays simulation
            n_steps = int(Config.NumericalParams.MAX_TIME/time_step) + 1
            times = np.zeros(n_steps)
            temps = np.zeros(n_steps)
            temps[0] = temp_initial
            powers = np.zeros(n_steps)
            power_available = np.zeros(n_steps)
            power_losses = np.zeros(n_steps)
            temp_local = np.zeros(n_steps)
            temp_surface = np.zeros(n_steps)
            h_convs = np.zeros(n_steps)

            evolution_data = [] if detailed else None
            
            # Simulation pas à pas
            for i in range(1, n_steps):
                times[i] = i * time_step
                
                try:
                    # Calcul chauffage avec détails
                    heating = heater.calculate_heating_power(
                        temp_fluid=temps[i-1],
                        voltage=voltage,
                        detailed=True
                    )
                    
                    # Vérification résultats valides
                    if not isinstance(heating, dict) or 'power' not in heating:
                        raise ValueError("Résultats heating_power invalides")
                    
                    powers[i] = heating['power']
                    power_available[i] = heating.get('power_available', heating['power'])
                    power_losses[i] = heating.get('power_losses', 0)
                    temp_local[i] = heating.get('temp_local', temps[i-1])
                    temp_surface[i] = heating['temp_surface']
                    h_convs[i] = heating.get('h_conv', 0)
                    
                    # Evolution température avec pertes
                    evolution = self.calculate_temperature_evolution(
                        temp_initial=temps[i-1],
                        temp_ambient=20.0,
                        power_heating=powers[i]*1000,  # kW -> W
                        time_step=time_step,
                        flow_rate=0
                    )
                    
                    temps[i] = evolution['temp']
                    
                    if detailed:
                        evolution_data.append({
                            'time': times[i],
                            'temperature': temps[i],
                            'heating': heating,
                            'evolution': evolution
                        })
                    
                    # Arrêt si température atteinte
                    if temps[i] >= temp_target:
                        idx = i+1
                        times = times[:idx]
                        temps = temps[:idx]
                        powers = powers[:idx]
                        power_available = power_available[:idx]
                        power_losses = power_losses[:idx]
                        temp_local = temp_local[:idx]
                        temp_surface = temp_surface[:idx]
                        h_convs = h_convs[:idx]
                        break
                        
                except Exception as e:
                    raise ValueError(f"Erreur à l'étape {i}: {str(e)}")
            
            # Résultats avec tous les paramètres
            results = {
                'times': times/60,  # conversion en minutes
                'temperatures': temps,
                'powers': powers,
                'power_available': power_available,
                'power_losses': power_losses,
                'temp_local': temp_local,
                'temp_surface': temp_surface,
                'h_conv': h_convs,
                'duration': times[-1]/60,
                'final_temp': temps[-1],
                'average_power': np.mean(powers),
                'total_energy': np.sum(powers) * time_step/3600  # kWh
            }

            if detailed:
                results['evolution_data'] = evolution_data
                    
            return results
            
        except Exception as e:
            raise ValueError(f"Erreur simulation chauffage: {str(e)}")

    def calculate_heating_cycles(
        self,
        heater: ElectricalHeater,
        temp_setpoint: float,  # °C
        temp_hysteresis: float = 2.0,  # °C
        duration: float = 3600,  # s
        time_step: float = 30,  # s
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Simule les cycles de chauffe avec hystérésis
        Utile pour dimensionner la régulation
        """
        # ... à implémenter si nécessaire ...
        pass


    