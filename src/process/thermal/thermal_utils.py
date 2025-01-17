# src/process/thermal/thermal_utils.py
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Union
import CoolProp.CoolProp as CP
from src.config import Config

# Configuration du logging
logger = logging.getLogger(__name__)

class ThermalProperties:
    """
    Calculs des propriétés thermiques et nombres adimensionnels
    Classe utilitaire pour tous les calculs thermiques du projet
    - Propriétés fluides (eau, vapeur)
    - Nombres adimensionnels (Re, Pr, Nu, etc.)
    - Coefficients de transfert thermique
    """
    
    @staticmethod
    def get_water_properties(
        temp: float,
        pressure: float = 1.0,
        raise_errors: bool = True,
        context: str = "process"
    ) -> Dict[str, float]:
        """
        Calcule les propriétés de l'eau avec protection contre les valeurs extrêmes
        """
        try:
            # Protection contre les températures extrêmes pour les calculs
            temp_calc = np.clip(temp, 
                            Config.ProcessLimits.TEMP_MIN - 10,  
                            Config.ProcessLimits.TEMP_MAX + 10)

            # Validation seulement si raise_errors est True
            if raise_errors:
                if context == "process":
                    Config.validate_operating_conditions(temp, pressure)
                elif context == "exchanger":
                    if not Config.ProcessLimits.TEMP_MIN <= temp <= Config.ProcessLimits.TEMP_STEAM_MAX:
                        raise ValueError(f"Température échangeur {temp}°C hors limites")

            # Conversion unités pour les calculs - utiliser temp_calc!
            T_K = 273.15 + temp_calc
            P_Pa = pressure * 1e5

            try:
                # Calcul propriétés avec la température protégée
                props = {
                    'rho': CP.PropsSI('D', 'T', T_K, 'P', P_Pa, 'Water'),
                    'cp': CP.PropsSI('C', 'T', T_K, 'P', P_Pa, 'Water'),
                    'k': CP.PropsSI('L', 'T', T_K, 'P', P_Pa, 'Water'),
                    'mu': CP.PropsSI('V', 'T', T_K, 'P', P_Pa, 'Water'),
                    'beta': CP.PropsSI('isobaric_expansion_coefficient', 'T', T_K, 'P', P_Pa, 'Water'),
                    'Pr': CP.PropsSI('Prandtl', 'T', T_K, 'P', P_Pa, 'Water')
                }

                # Vérification et ajout des propriétés dérivées
                for key, value in props.items():
                    if not Config.NumericalParams.MIN_VALUE <= abs(value) <= Config.NumericalParams.MAX_VALUE:
                        if raise_errors:
                            raise ValueError(f"Propriété {key} hors limites: {value}")
                        else:
                            props[key] = Config.NumericalParams.MIN_VALUE

                # Calculs dérivés
                props['nu'] = props['mu'] / props['rho']
                props['alpha'] = props['k'] / (props['rho'] * props['cp'])

                return props

            except Exception as e:
                if raise_errors:
                    raise
                # Valeurs par défaut en cas d'erreur
                return {
                    'rho': Config.PhysicalConstants.RHO_WATER,
                    'cp': Config.PhysicalConstants.CP_WATER,
                    'mu': Config.PhysicalConstants.MU_WATER,
                    'k': Config.PhysicalConstants.K_WATER,
                    'beta': Config.PhysicalConstants.BETA_WATER,
                    'Pr': Config.PhysicalConstants.CP_WATER * Config.PhysicalConstants.MU_WATER / Config.PhysicalConstants.K_WATER,
                    'nu': Config.PhysicalConstants.MU_WATER / Config.PhysicalConstants.RHO_WATER,
                    'alpha': Config.PhysicalConstants.K_WATER / (Config.PhysicalConstants.RHO_WATER * Config.PhysicalConstants.CP_WATER)
                }

        except Exception as e:
            if raise_errors:
                raise
            logger.error(f"Erreur calcul propriétés: {str(e)}")
            return None

    @staticmethod
    def get_steam_properties(
        pressure: float,
        raise_errors: bool = True
    ) -> Dict[str, float]:
        """
        Calcule les propriétés vapeur saturée
        Args:
            pressure: pression en bar
            raise_errors: si True, lève les erreurs
        Returns:
            dict avec températures et enthalpies
        """
        try:
            Config.validate_operating_conditions(
                Config.ProcessLimits.TEMP_MAX, pressure
            )
            
            P_Pa = pressure * 1e5  # bar -> Pa
            
            try:
                # Calcul propriétés
                T_sat = CP.PropsSI('T', 'P', P_Pa, 'Q', 1, 'Water')
                h_vap = CP.PropsSI('H', 'P', P_Pa, 'Q', 1, 'Water')  # vapeur
                h_liq = CP.PropsSI('H', 'P', P_Pa, 'Q', 0, 'Water')  # liquide
                cp_vap = CP.PropsSI('C', 'P', P_Pa, 'Q', 1, 'Water')
                rho_vap = CP.PropsSI('D', 'P', P_Pa, 'Q', 1, 'Water')
                
                return {
                    'temperature': T_sat - 273.15,  # K -> °C
                    'h_vapor': h_vap/1000,         # J/kg -> kJ/kg
                    'h_liquid': h_liq/1000,        # J/kg -> kJ/kg
                    'h_vaporization': (h_vap - h_liq)/1000,
                    'cp': cp_vap,                  # J/kg.K
                    'rho': rho_vap                 # kg/m³
                }
                
            except Exception as e:
                if raise_errors:
                    raise ValueError(f"Erreur CoolProp vapeur: {str(e)}")
                else:
                    logger.warning(
                        f"Utilisation approximation vapeur saturée à {pressure}bar"
                    )
                    # Approximations à pression modérée
                    t_sat = 100 + 20 * (pressure - 1)  # °C, approximatif
                    h_vap = 2260  # kJ/kg, approximatif
                    return {
                        'temperature': t_sat,
                        'h_vapor': h_vap + 2000,
                        'h_liquid': 420,
                        'h_vaporization': h_vap,
                        'cp': 2000,
                        'rho': 1
                    }
                    
        except Exception as e:
            if raise_errors:
                raise
            else:
                logger.error(f"Erreur calcul vapeur: {str(e)}")
                return None

    @staticmethod
    def calculate_reynolds(
        velocity: float,  # m/s
        length: float,    # m
        temp: float,      # °C
        pressure: float = 1.0,  # bar
        context: str = ""
    ) -> float:
        """
        Calcule le nombre de Reynolds avec validation
        Args:
            context: information pour les warnings
        """
        # Évite division par zéro
        min_vel = Config.NumericalParams.MIN_VALUE
        velocity = max(abs(velocity), min_vel)
        
        props = ThermalProperties.get_water_properties(temp, pressure)
        re = velocity * length / props['nu']
        
        # Validation selon régime
        if re < Config.CorrelationLimits.RE_MIN:
            logger.warning(
                f"Reynolds très faible ({re:.1f}) pour {context}: "
                f"convection négligeable"
            )
        elif Config.CorrelationLimits.RE_MAX_LAMINAR < re < Config.CorrelationLimits.RE_MIN_TURB:
            logger.info(
                f"Reynolds en zone transitoire ({re:.0f}) pour {context}"
            )
        elif re > Config.CorrelationLimits.RE_MAX_TURB:
            logger.warning(
                f"Reynolds très élevé ({re:.1e}) pour {context}"
            )
            
        return re

    @staticmethod
    def calculate_grashof(
        temp_surf: float,    # °C
        temp_fluid: float,   # °C
        length: float,       # m
        pressure: float = 1.0,  # bar
        temp_film: Optional[float] = None  # °C
    ) -> float:
        """
        Calcule le nombre de Grashof
        Args:
            temp_film: température du film (moyenne si non spécifiée)
        """
        if temp_film is None:
            temp_film = (temp_surf + temp_fluid) / 2
            
        props = ThermalProperties.get_water_properties(temp_film, pressure)
        g = Config.PhysicalConstants.G
        delta_t = abs(temp_surf - temp_fluid)
        
        gr = g * props['beta'] * delta_t * length**3 / props['nu']**2
        
        if gr > Config.CorrelationLimits.GR_MAX:
            logger.warning(
                f"Grashof très élevé ({gr:.1e}): "
                f"convection naturelle très forte"
            )
            
        return min(gr, Config.CorrelationLimits.GR_MAX)

    @staticmethod
    def calculate_nusselt_forced(
        reynolds: float,
        prandtl: float,
        geometry: str = 'pipe'
    ) -> float:
        """
        Calcule Nusselt en convection forcée
        
        Corrélations utilisées :
        - Tube : Gnielinski pour Re > 4000
        - Plaque : correlation de plaque plane
        
        La transition laminaire/turbulent utilise
        une interpolation pour éviter discontinuité
        """
        def gnielinski(re: float, pr: float) -> float:
            """Correlation de Gnielinski pour tubes"""
            f = (0.79 * np.log(re) - 1.64)**-2  # Facteur friction
            return (f/8 * (re-1000) * pr) / \
                   (1 + 12.7 * (f/8)**0.5 * (pr**(2/3) - 1))
                   
        if geometry == 'pipe':
            if reynolds > Config.CorrelationLimits.RE_MIN_TURB:
                # Turbulent
                return gnielinski(reynolds, prandtl)
                
            elif reynolds > Config.CorrelationLimits.RE_MAX_LAMINAR:
                # Zone transitoire - interpolation
                nu_lam = 3.66  # Laminaire établi
                nu_turb = gnielinski(
                    Config.CorrelationLimits.RE_MIN_TURB, 
                    prandtl
                )
                # Interpolation avec tangente hyperbolique
                x = (reynolds - Config.CorrelationLimits.RE_MAX_LAMINAR) / \
                    (Config.CorrelationLimits.RE_MIN_TURB - 
                     Config.CorrelationLimits.RE_MAX_LAMINAR)
                blend = 0.5 * (1 + np.tanh(2*np.pi*(x - 0.5)))
                return nu_lam * (1-blend) + nu_turb * blend
            else:
                # Laminaire
                return max(3.66, Config.CorrelationLimits.NU_MIN)
                
        else:  # Plaque plane
            if reynolds > 5e5:
                # Turbulent
                return 0.037 * reynolds**0.8 * prandtl**(1/3)
            else:
                # Laminaire
                return max(
                    0.664 * reynolds**0.5 * prandtl**(1/3),
                    Config.CorrelationLimits.NU_MIN
                )

    @staticmethod
    def calculate_nusselt_natural(
        grashof: float,
        prandtl: float,
        geometry: str = 'vertical'
    ) -> float:
        """
        Calcule Nusselt en convection naturelle
        selon corrélations validées
        
        Corrélations :
        - Verticale : Churchill & Chu
        - Horizontale : McAdams
        """
        ra = grashof * prandtl  # Nombre de Rayleigh
        
        if geometry == 'vertical':
            # Churchill & Chu
            if ra < 1e9:  # Laminaire
                nu = 0.68 + 0.67 * ra**0.25 / \
                     (1 + (0.492/prandtl)**(9/16))**(4/9)
            else:  # Turbulent
                nu = (0.825 + 0.387 * ra**(1/6) / \
                     (1 + (0.492/prandtl)**(9/16))**(8/27))**2
                     
        elif geometry == 'horizontal_up':
            # Surface chaude vers le haut
            if ra < 1e7:  # Laminaire
                nu = 0.54 * ra**0.25
            else:  # Turbulent
                nu = 0.15 * ra**(1/3)
                
        else:  # horizontal_down
            # Surface chaude vers le bas
            nu = 0.27 * ra**0.25
            
        return max(nu, Config.CorrelationLimits.NU_MIN)

    @staticmethod
    def calculate_convection_coefficient(
        temp_surf: float,     # °C
        temp_fluid: float,    # °C
        length: float,        # m
        velocity: float = 0,  # m/s
        pressure: float = 1.0,  # bar
        geometry: str = 'vertical'
    ) -> Dict[str, float]:
        """
        Calcule le coefficient de convection total
        avec prise en compte convection mixte
        Returns:
            dict avec h et paramètres du calcul
        """
        # Température moyenne du film
        temp_film = (temp_surf + temp_fluid) / 2
        props = ThermalProperties.get_water_properties(temp_film, pressure)
        
        results = {'properties': props}
        
        # Convection forcée si vitesse non nulle
        if velocity > Config.NumericalParams.MIN_VALUE:
            re = ThermalProperties.calculate_reynolds(
                velocity, length, temp_film, pressure,
                context=f"convection {geometry}"
            )
            nu_forced = ThermalProperties.calculate_nusselt_forced(
                re, props['Pr'], geometry
            )
            h_forced = nu_forced * props['k'] / length
            results.update({
                'reynolds': re,
                'nu_forced': nu_forced,
                'h_forced': h_forced
            })
        else:
            h_forced = 0
            
        # Convection naturelle
        gr = ThermalProperties.calculate_grashof(
            temp_surf, temp_fluid, length, pressure, temp_film
        )
        nu_natural = ThermalProperties.calculate_nusselt_natural(
            gr, props['Pr'], geometry
        )
        h_natural = nu_natural * props['k'] / length
        results.update({
            'grashof': gr,
            'nu_natural': nu_natural,
            'h_natural': h_natural
        })
        
        # Combinaison des effets selon la géométrie
        if geometry == 'vertical':
            # Méthode de Churchill pour convection mixte verticale
            h_total = (abs(h_forced**3) + abs(h_natural**3))**(1/3)
        else:
            # Pour flux horizontal, on prend le max
            h_total = max(h_forced, h_natural)
            
        # Application facteur de sécurité
        h_total *= Config.SafetyFactors.HEAT_TRANSFER
        
        # Validation finale
        if h_total < Config.CorrelationLimits.MIN_HEAT_TRANSFER:
            logger.warning(
                f"Coefficient h très faible ({h_total:.1f} W/m².K) "
                f"pour {geometry}"
            )
            h_total = Config.CorrelationLimits.MIN_HEAT_TRANSFER
            
        elif h_total > Config.CorrelationLimits.MAX_HEAT_TRANSFER:
            logger.warning(
                f"Coefficient h très élevé ({h_total:.1f} W/m².K) "
                f"pour {geometry}"
            )
            h_total = Config.CorrelationLimits.MAX_HEAT_TRANSFER
            
        results['h_total'] = h_total
        return results

    @staticmethod
    def calculate_heat_exchanger_ntu(
        flow_rate: float,     # m³/h
        surface: float,       # m²
        temp_hot: float,      # °C
        temp_cold: float,     # °C
        pressure: float = 1.0,  # bar
        k: float = 3000,      # W/m².K
        fouling: float = 0.9  # facteur d'encrassement
    ) -> Tuple[float, float]:
        """
        Calcule l'efficacité de l'échangeur par méthode NUT
        avec prise en compte des conditions réelles
        Returns:
            efficacité [-], coefficient d'échange global [W/m².K]
        """
        # Logs début calcul
        logger.info("\nDébut calcul échangeur NUT:")
        logger.info(f"- Températures: froide={temp_cold:.1f}°C, chaude={temp_hot:.1f}°C")
                
        # Propriétés à température moyenne  
        temp_mean = (temp_hot + temp_cold) / 2
        logger.info(f"- Température moyenne: {temp_mean:.1f}°C")

        props = ThermalProperties.get_water_properties(
            temp_mean, 
            pressure,
            context="exchanger"
        )
        logger.info(f"- Propriétés eau: rho={props['rho']:.1f} kg/m³, cp={props['cp']:.1f} J/kg.K")
                
        # Débit massique
        mdot = flow_rate * props['rho'] / 3600  # kg/s
        logger.info(f"- Débit massique: {mdot:.3f} kg/s")
                
        # Capacité thermique minimale
        Cmin = mdot * props['cp']  # W/K
        logger.info(f"- Capacité thermique: {Cmin:.1f} W/K")
                
        # Coefficient global avec facteur de sécurité et encrassement
        UA = k * surface * fouling * Config.SafetyFactors.HEAT_TRANSFER  # W/K
        logger.info(f"- Coefficient UA: {UA:.1f} W/K")
                
        # NUT et efficacité
        NUT = UA / Cmin
        logger.info(f"- NUT: {NUT:.2f}")
        
        efficiency = (1 - np.exp(-NUT)) / (1 + np.exp(-NUT))
        logger.info(f"- Efficacité finale: {efficiency:.2f}")
                
        # Vérification performance
        if efficiency < 0.6:
            logger.warning(
                f"Efficacité échangeur faible ({efficiency:.2f}): "
                f"vérifier dimensionnement"
            )
                    
        if efficiency > 0.95:
            logger.info(
                f"Très bonne efficacité échangeur ({efficiency:.2f})"
            )
                
        return efficiency, UA
    
    @staticmethod
    def validate_process_temperature(
        temp: float, 
        with_margin: bool = True,
        context: str = "process"
    ) -> float:
        """
        Valide et ajuste une température process
        Args:
            temp: Température à valider
            with_margin: Si True, ajoute une marge de sécurité
            context: 'process' ou 'exchanger' pour différentes limites
        Returns:
            Température validée et ajustée
        """
        margin = 1.0 if with_margin else 0.0
        
        if context == "process":
            temp_max = Config.ProcessLimits.TEMP_MAX - margin
        else:  # context == "exchanger"
            temp_max = Config.ProcessLimits.TEMP_STEAM_MAX - margin
            
        return np.clip(
            temp,
            Config.ProcessLimits.TEMP_MIN,
            temp_max
        )

    @staticmethod
    def validate_temperature_range(
        temp_in: float,
        temp_out: float,
        context: str = "process"
    ) -> Tuple[float, float]:
        """
        Valide une plage de températures
        Returns:
            Tuple (temp_in validée, temp_out validée)
        """
        temp_in_safe = ThermalProperties.validate_process_temperature(
            temp_in, context=context
        )
        temp_out_safe = ThermalProperties.validate_process_temperature(
            temp_out, context=context
        )
        return temp_in_safe, temp_out_safe