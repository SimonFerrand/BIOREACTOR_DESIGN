# src/equipment/heat_system/electrical_heater.py
from dataclasses import dataclass
from typing import Dict, Optional, Any, List  
import numpy as np
import logging
import time
from src.config import Config
from src.process.thermal.thermal_utils import ThermalProperties

# Configuration du logging
logger = logging.getLogger(__name__)

@dataclass
class ElectricalHeaterSpecs:
    """Spécifications de la résistance chauffante"""
    power: float          # kW
    surface: float        # m² (surface d'échange)
    max_temp: float       # °C (température max résistance)
    power_density: float  # kW/m² (densité de puissance surfacique)
    voltage: float = Config.ElectricalHeaterLimits.MIN_VOLTAGE
    material: str = "Incoloy800"
    
    def __post_init__(self):
        # Validation puissance selon Config
        if not Config.ElectricalHeaterLimits.MIN_POWER <= self.power <= Config.ElectricalHeaterLimits.MAX_POWER:
            raise ValueError(
                f"Puissance {self.power}kW hors limites "
                f"({Config.ElectricalHeaterLimits.MIN_POWER}-{Config.ElectricalHeaterLimits.MAX_POWER}kW)"
            )
            
        # Validation température max selon matériau
        material_max_temp = Config.ElectricalHeaterLimits.MATERIALS[self.material]['max_temp']
        if self.max_temp > material_max_temp:
            raise ValueError(
                f"Température max {self.max_temp}°C > limite matériau {material_max_temp}°C"
            )
        
        # Validation densité de puissance
        actual_density = self.power / self.surface
        if not Config.ElectricalHeaterLimits.MIN_POWER_DENSITY <= actual_density <= Config.ElectricalHeaterLimits.MAX_POWER_DENSITY:
            raise ValueError(
                f"Densité puissance {actual_density:.1f} kW/m² hors limites "
                f"({Config.ElectricalHeaterLimits.MIN_POWER_DENSITY}-{Config.ElectricalHeaterLimits.MAX_POWER_DENSITY}kW/m²)"
            )

class ElectricalHeater:
    """Système de chauffe par résistance électrique"""
    def __init__(
        self,
        power: float,           # kW
        surface: float,         # m²  
        max_temp: float = Config.ProcessLimits.TEMP_MAX, 
        power_density: float = Config.ElectricalHeaterLimits.MAX_POWER_DENSITY/2,
        material: str = "Incoloy800"
    ):
        """
        Args:
            power: Puissance nominale en kW
            surface: Surface d'échange en m²
            max_temp: Température maximale de la résistance
            power_density: Densité de puissance maximale
            material: Matériau de la résistance
        """
        self.specs = ElectricalHeaterSpecs(
            power=power,
            surface=surface, 
            max_temp=max_temp,
            power_density=power_density,
            material=material
        )

        # Variables PID standard
        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_time = 0.0
        
        # Nouvelles variables pour inertie et filtrage
        self.last_power = 0.0
        self.last_temp = None
        self.alpha_temp = 0.2  # Constante filtrage température
        self.thermal_mass = self.specs.surface * 50  # J/K (approximation)
        
    def calculate_heating_power(
        self,
        temp_fluid: float,      # °C
        temp_ambient: float = 20.0,
        temp_surface: Optional[float] = None,  # °C
        voltage: Optional[float] = None,  # V
        height_ratio: float = 0.5,  # Position relative dans le tank
        detailed: bool = False
    ) -> Dict[str, float]:
        """
        Calcule puissance réelle selon conditions avec stratification
        avec modèle thermique rigoureux
        """
        try:
            # 1. Validation des conditions opératoires
            Config.validate_operating_conditions(temp_fluid, 1.0)
            Config.validate_operating_conditions(temp_ambient, 1.0)
            
            # 2. Calcul température locale avec stratification
            temp_local = self._calculate_stratification(
                temp_fluid=temp_fluid,
                height_ratio=height_ratio,
                power=self.specs.power
            )
            
            if temp_local >= self.specs.max_temp:
                return {'power': 0} if not detailed else {
                    'power': 0,
                    'limiting_factor': 'temperature',
                    'temp_local': temp_local
                }
            
            # 3. Puissance disponible avec sécurités
            power_available = self.specs.power * Config.ElectricalHeaterLimits.POWER_SAFETY * 1000  # kW -> W
            
            if voltage is not None:
                if not (Config.ElectricalHeaterLimits.MIN_VOLTAGE <= voltage <= 
                    Config.ElectricalHeaterLimits.MAX_VOLTAGE):
                    raise ValueError(
                        f"Tension {voltage}V hors limites "
                        f"({Config.ElectricalHeaterLimits.MIN_VOLTAGE}-"
                        f"{Config.ElectricalHeaterLimits.MAX_VOLTAGE}V)"
                    )
                power_available *= (voltage/self.specs.voltage) ** 2
            
            # 4. Calcul coefficients d'échange
            h_conv = self._calculate_heat_transfer_coeff(
                temp_fluid=temp_local,
                temp_surface=temp_surface,
                pressure=1.0
            )
            
            h_ext = max(10.0, Config.CorrelationLimits.MIN_HEAT_TRANSFER)  # W/m²K pour l'air
            
            # 5. Surface effective avec facteur sécurité
            surface_effective = self.specs.surface * Config.SafetyFactors.HEAT_TRANSFER
            
            # 6. Résistances thermiques
            # Convection interne
            R_conv_int = 1 / (h_conv * surface_effective)  # K/W
            
            # Conduction paroi (matériau résistance)
            e_wall = 0.003  # m (3mm standard)
            k_wall = Config.ElectricalHeaterLimits.MATERIALS[self.specs.material]['conductivity']
            R_wall = e_wall / (k_wall * surface_effective)  # K/W
            
            # Isolation avec résistance contact
            e_insulation = 0.025  # m (25mm standard)
            k_insulation = Config.MaterialProperties.K_MINERAL_WOOL
            R_contact = Config.MaterialProperties.THERMAL_CONTACT['316L']['mineral_wool']
            
            R_insulation = (e_insulation / (k_insulation * surface_effective) + 
                        R_contact / surface_effective)  # K/W
            
            # Convection externe
            R_conv_ext = 1 / (h_ext * surface_effective)  # K/W
            
            # Résistance totale
            R_total = R_conv_int + R_wall + R_insulation + R_conv_ext
            
            # 7. Calcul flux thermique et température surface
            delta_t = temp_local - temp_ambient
            
            # Flux avec limitation par température surface
            flux_max = power_available / surface_effective  # W/m²
            
            if temp_surface is None:
                # Protection surchauffe
                delta_t_max = (self.specs.max_temp - temp_local) * \
                            Config.ElectricalHeaterLimits.TEMP_SAFETY
                flux_max = min(flux_max, h_conv * delta_t_max)
                temp_surface = temp_local + flux_max/h_conv
                
            # 8. Calcul puissances
            # Pertes thermiques
            q_loss = delta_t / R_total  # W
            
            # Facteur correctif non-linéaire pour petits delta T
            f_corr = min(1.0, (delta_t/30)**0.5)  # Correction empirique
            q_loss *= f_corr
            
            # Protection contre pertes excessives
            q_loss = min(q_loss, 0.25 * power_available)  # Max 25% de pertes
            
            # Puissance effective
            power_effective = max(0, power_available - q_loss)
            
            # 9. Résultats
            results = {
                'power': power_effective/1000,  # W -> kW
                'temp_surface': temp_surface,
                'temp_local': temp_local
            }
            
            if detailed:
                results.update({
                    'h_conv': h_conv,
                    'h_ext': h_ext,
                    'flux': flux_max,
                    'power_available': power_available/1000,  # W -> kW
                    'power_losses': q_loss/1000,  # W -> kW
                    'R_total': R_total,
                    'delta_t': delta_t,
                    'height_ratio': height_ratio,
                    'surface_effective': surface_effective,
                    'thermal_efficiency': (power_effective/power_available * 100)
                })
                
            # Log des résultats
            logger.info(
                f"Heating power calculation:"
                f"\n- Température locale: {temp_local:.1f}°C"
                f"\n- Puissance disponible: {power_available/1000:.1f}kW"
                f"\n- Pertes: {q_loss/1000:.1f}kW ({q_loss/power_available*100:.1f}%)"
                f"\n- Coefficient global: {1/(R_total*surface_effective):.0f}W/m²K"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur calcul puissance: {str(e)}")
            raise
    
    def _calculate_heat_transfer_coeff(
        self,
        temp_fluid: float,
        temp_surface: Optional[float] = None,
        pressure: float = 1.0
    ) -> float:
        """
        Calcule coefficient convection naturelle (Churchill & Chu)
        """
        try:
            # Protection températures invalides
            if temp_fluid < Config.ProcessLimits.TEMP_MIN:
                raise ValueError(f"Température fluide {temp_fluid}°C < minimum")
                
            if temp_surface is not None and temp_surface < temp_fluid:
                logger.warning(
                    f"Température surface {temp_surface}°C < fluide {temp_fluid}°C"
                )
                temp_surface = None
            
            # Température moyenne pour propriétés
            temp_mean = temp_surface if temp_surface else temp_fluid + 5
            temp_film = (temp_mean + temp_fluid) / 2
            
            # Propriétés physiques
            props = ThermalProperties.get_water_properties(
                temp_film, pressure, context="process"
            )
            
            # Nombres adimensionnels avec protection
            g = Config.PhysicalConstants.G
            beta = props['beta']
            delta_t = abs(temp_mean - temp_fluid)
            l_carac = (4 * self.specs.surface / np.pi) ** 0.5
            
            # Protection valeurs nulles
            nu = max(props['nu'], Config.NumericalParams.MIN_VALUE)
            
            gr = g * beta * delta_t * l_carac**3 / nu**2
            gr = min(gr, Config.CorrelationLimits.GR_MAX)
            
            pr = min(max(props['Pr'], Config.CorrelationLimits.PR_MIN),
                    Config.CorrelationLimits.PR_MAX)
            
            ra = gr * pr
            
            # Nusselt selon régime
            if ra < 1e9:  # Laminaire
                nu = 0.68 + 0.67 * ra**0.25 / \
                    (1 + (0.492/pr)**(9/16))**(4/9)
            else:  # Turbulent
                nu = (0.825 + 0.387 * ra**(1/6) / \
                    (1 + (0.492/pr)**(9/16))**(8/27))**2
                
            nu = max(nu, Config.CorrelationLimits.NU_MIN)
            
            # Coefficient final avec sécurité
            h = nu * props['k'] / l_carac * Config.SafetyFactors.HEAT_TRANSFER
            
            return max(h, Config.CorrelationLimits.MIN_HEAT_TRANSFER)
            
        except Exception as e:
            logger.error(f"Erreur calcul coefficient transfert: {str(e)}")
            return Config.CorrelationLimits.MIN_HEAT_TRANSFER
    
    def _calculate_stratification(
        self,
        temp_fluid: float,
        height_ratio: float,
        power: float
    ) -> float:
        """
        Calcule stratification thermique selon puissance
        """
        try:
            # Stratification limitée selon Config
            strat_factor = min(
                Config.ElectricalHeaterLimits.STRATIFICATION_FACTOR,
                power / self.specs.power
            )
            
            # Température locale avec gradient max
            delta_t_max = Config.CorrelationLimits.MAX_TEMP_GRADIENT
            temp_local = temp_fluid * (1 + strat_factor * (height_ratio - 0.5))
            
            # Protection température max
            return min(temp_local, self.specs.max_temp)
            
        except Exception as e:
            logger.error(f"Erreur calcul stratification: {str(e)}")
            return temp_fluid

    def analyze_performance(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyse performances du système de chauffe
        Args:
            results: Dictionnaire des résultats de simulation
        Returns:
            Dictionnaire des métriques de performance
        """
        try:
            # Protection données manquantes
            required_keys = ['times', 'powers', 'power_available', 'power_losses']
            if not all(key in results for key in required_keys):
                raise ValueError(
                    f"Résultats incomplets. Requis: {required_keys}"
                )
            
            # Conversion en arrays numpy
            times = np.array(results['times'])
            temperatures = np.array(results['temperatures'])
            power_available = np.array(results['power_available'])
            power_out = np.array(results['powers'])
            power_losses = np.array(results['power_losses'])
            
            # Protection division par zéro
            mask = power_available > Config.NumericalParams.MIN_VALUE
            
            if not np.any(mask):
                raise ValueError("Aucune puissance disponible détectée")
            
            # Calcul efficacité et pertes où puissance > 0
            thermal_efficiency = np.mean(power_out[mask] / power_available[mask]) * 100
            heat_loss_ratio = np.mean(power_losses[mask] / power_available[mask]) * 100
            
            # Taux de chauffe moyen
            times_sec = times * 60  # minutes -> secondes
            temp_diff = temperatures[-1] - temperatures[0]
            time_diff = (times_sec[-1] - times_sec[0]) / 60  # Retour en minutes
            heating_rate = temp_diff / time_diff if time_diff > 0 else 0
            
            # Calcul stratification
            if 'temp_local' in results:
                temp_local = np.array(results['temp_local'])
                strat_max = np.max(np.abs(temp_local - temperatures))
            else:
                strat_max = 0
                
            # Calcul énergies (kWh)
            dt = np.diff(times_sec, prepend=times_sec[0]) / 3600  # s -> h
            energy_total = np.sum(power_available * dt)
            energy_useful = np.sum(power_out * dt)
            energy_lost = np.sum(power_losses * dt)
            
            # Coefficient d'échange moyen
            h_conv = [
                d['heating']['h_conv'] 
                for d in results['evolution_data'] 
                if 'heating' in d
            ]
            avg_htc = np.mean(h_conv) if h_conv else 0
            
            metrics = {
                'thermal_efficiency': thermal_efficiency,
                'heat_loss_ratio': heat_loss_ratio,
                'max_surface_temp': np.max(results['temp_surface']),
                'heating_rate': heating_rate,
                'stratification_max': strat_max,
                'energy_total': energy_total,
                'energy_useful': energy_useful,
                'energy_lost': energy_lost,
                'average_htc': avg_htc
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur analyse performances: {str(e)}")
            raise

    def validate_performance(
        self,
        metrics: Dict[str, float]
    ) -> List[str]:
        """
        Valide les performances et retourne les avertissements
        Args:
            metrics: Dictionnaire des métriques calculées
        Returns:
            Liste des avertissements
        """
        warnings = []
        
        # Efficacité minimale selon sécurité config
        min_efficiency = Config.ElectricalHeaterLimits.POWER_SAFETY * 100
        if metrics['thermal_efficiency'] < min_efficiency:
            warnings.append(
                f"Efficacité thermique {metrics['thermal_efficiency']:.1f}% < "
                f"minimum {min_efficiency}%"
            )
        
        # Pertes maximales selon sécurité
        max_losses = (1 - Config.ElectricalHeaterLimits.POWER_SAFETY) * 100
        if metrics['heat_loss_ratio'] > max_losses:
            warnings.append(
                f"Pertes thermiques {metrics['heat_loss_ratio']:.1f}% > "
                f"maximum {max_losses}%"
            )
        
        # Taux de chauffe minimum
        if metrics['heating_rate'] < 1:
            warnings.append(
                f"Taux de chauffe {metrics['heating_rate']:.2f}°C/min trop faible"
            )
        
        # Stratification maximale
        max_strat = Config.CorrelationLimits.MAX_TEMP_GRADIENT
        if metrics['stratification_max'] > max_strat:
            warnings.append(
                f"Stratification {metrics['stratification_max']:.1f}°C > "
                f"maximum {max_strat}°C"
            )
            
        return warnings
    
    # Dans ElectricalHeater
    def calculate_heating_power_with_control(
            self,
            temp_fluid: float,     
            temp_target: float,    
            temp_measured: float,  
            margin: float = Config.ProcessLimits.CIP_TEMP_DEADBAND,  
            time_step: float = 1.0,
            detailed: bool = False
        ) -> Dict[str, float]:
            try:
                # 1. Calcul température locale avec stratification
                temp_local = self._calculate_stratification(
                    temp_fluid=temp_fluid,
                    height_ratio=0.5,
                    power=self.specs.power
                )

                # 2. Protection température avec hystérésis
                safety_margin = 2.0  # °C
                if temp_local >= (self.specs.max_temp - safety_margin) * Config.ElectricalHeaterLimits.TEMP_SAFETY:
                    return {'power': 0} if not detailed else {
                        'power': 0,
                        'control_info': {
                            'error': 0,
                            'power_factor': 0,
                            'temp_local': temp_local,
                            'limiting_factor': 'temperature'
                        }
                    }

                # 3. Filtrage température et calcul erreur
                if self.last_temp is not None:
                    temp_filtered = (1 - self.alpha_temp) * self.last_temp + self.alpha_temp * temp_measured
                else:
                    temp_filtered = temp_measured
                self.last_temp = temp_filtered
                
                error = temp_target - temp_filtered
                
                # Zone morte pour éviter oscillations
                if abs(error) < margin/2:
                    error = 0

                # 4. Contrôle PID avec anti-windup amélioré
                kp = 0.3 * Config.SafetyFactors.THERMAL
                ki = 0.05 * Config.SafetyFactors.THERMAL
                kd = 0.02 * Config.SafetyFactors.THERMAL

                # Terme proportionnel avec saturation douce
                if abs(error) < margin:
                    p_term = kp * (error/margin) * np.exp(-abs(error)/margin)
                else:
                    p_term = kp * np.sign(error)

                # Terme intégral avec anti-windup et reset
                if abs(error) > 3*margin:  # Reset si erreur trop grande
                    self.error_integral = 0
                else:
                    self.error_integral = np.clip(
                        self.error_integral + error * time_step,
                        -self.specs.power/(2*ki),  # Limites plus strictes
                        self.specs.power/(2*ki)
                    )
                i_term = ki * self.error_integral

                # Terme dérivé avec filtrage
                if hasattr(self, 'last_error'):
                    d_term = kd * (error - self.last_error) / time_step
                else:
                    d_term = 0
                self.last_error = error

                # 5. Calcul puissance avec rampe et inertie
                power_raw = p_term + i_term + d_term
                
                # Limitation du taux de variation
                if hasattr(self, 'last_power'):
                    max_change = 0.2 * time_step  # 20% max par seconde
                    power_delta = np.clip(
                        power_raw - self.last_power,
                        -max_change,
                        max_change
                    )
                    power_factor = np.clip(
                        self.last_power + power_delta,
                        0.1,
                        1.0
                    )
                else:
                    power_factor = np.clip(power_raw, 0.1, 1.0)
                
                # Calcul puissance avec sécurité
                power = power_factor * self.specs.power * Config.ElectricalHeaterLimits.POWER_SAFETY
                
                # Ajout inertie thermique
                if hasattr(self, 'last_power'):
                    tau = self.thermal_mass / 3600  # h
                    factor = np.exp(-time_step/tau)
                    power = factor * self.last_power + (1-factor) * power
                
                self.last_power = power

                # Log pour debug
                logger.info(
                    f"Control calculation:"
                    f"\n- Température mesurée: {temp_measured:.1f}°C"
                    f"\n- Température filtrée: {temp_filtered:.1f}°C"
                    f"\n- Erreur: {error:.2f}°C"
                    f"\n- Termes PID: P={p_term:.2f}, I={i_term:.2f}, D={d_term:.2f}"
                    f"\n- Puissance: {power:.1f}kW"
                )

                if detailed:
                    return {
                        'power': power,
                        'control_info': {
                            'temp_local': temp_local,
                            'temp_filtered': temp_filtered,
                            'error': error,
                            'power_factor': power_factor,
                            'p_term': p_term,
                            'i_term': i_term,
                            'd_term': d_term,
                            'power_raw': power_raw,
                            'thermal_factor': factor if hasattr(self, 'last_power') else 1.0
                        }
                    }
                return {'power': power}

            except Exception as e:
                logger.error(f"Erreur calcul puissance contrôle: {str(e)}")
                return {'power': 0.1 * self.specs.power}  # Puissance minimale sécurité
        
def create_standard_heater(power: float) -> ElectricalHeater:
    """
    Crée une résistance standard selon spécifications
    Args:
        power: Puissance en kW
    Returns:
        Instance de ElectricalHeater configurée
    Raises:
        ValueError si puissance non standard
    """
    # Validation puissance selon Config
    if not Config.ElectricalHeaterLimits.MIN_POWER <= power <= Config.ElectricalHeaterLimits.MAX_POWER:
        raise ValueError(
            f"Puissance {power}kW hors limites "
            f"({Config.ElectricalHeaterLimits.MIN_POWER}-{Config.ElectricalHeaterLimits.MAX_POWER}kW)"
        )
        
    # Standards industriels validés
    specs = {
        3.0: {'surface': 0.3, 'density': 10},   # 3 kW
        6.0: {'surface': 0.5, 'density': 12},   # 6 kW
        9.0: {'surface': 0.6, 'density': 15},    # 9 kW
        12.0: {'surface': 0.8, 'density': 15},   # 12 kW
        24.0: {'surface': 1.6, 'density': 15}    # 24 kW (2x12kW)
    }
    
    if power not in specs:
        raise ValueError(
            f"Puissance {power}kW non standard. "
            f"Valeurs possibles: {list(specs.keys())}kW"
        )
        
    spec = specs[power]
    
    # Validation densité de puissance
    density = spec['density']
    if not Config.ElectricalHeaterLimits.MIN_POWER_DENSITY <= density <= Config.ElectricalHeaterLimits.MAX_POWER_DENSITY:
        raise ValueError(
            f"Densité puissance {density}kW/m² hors limites "
            f"({Config.ElectricalHeaterLimits.MIN_POWER_DENSITY}-{Config.ElectricalHeaterLimits.MAX_POWER_DENSITY})"
        )
        
    # Utiliser la température process max comme limite opératoire
    return ElectricalHeater(
        power=power,
        surface=spec['surface'],
        power_density=density,
        max_temp=Config.ProcessLimits.TEMP_MAX,  # Changé ici
        material="Incoloy800"
    )