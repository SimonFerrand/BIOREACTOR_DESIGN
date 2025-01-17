# src/config.py
from typing import Dict

class Config:
    """Configuration globale du projet bioréacteur"""
    
    class ProcessLimits:
        """Limites opératoires process"""
        # Debit
        CIP_FLOW_MIN = 2.0    # m³/h
        CIP_FLOW_MAX = 6.0    # m³/h
        CIP_TEMP_DEADBAND = 2.0  # °C (hystérésis)

        # Températures
        TEMP_MIN = 5.0    # °C
        TEMP_MAX = 95.0   # °C
        
        # Pressions process
        PRESS_MIN = 0.0   # bar (minimum absolu)
        PRESS_MAX = 8.0   # bar (maximum CIP)

        # Températures vapeur
        TEMP_STEAM_MIN = 100.0  # °C
        TEMP_STEAM_MAX = 150.0  # °C
        
        # Pressions CIP (IM-10)
        CIP_PRESS_MIN = 4.0  # bar
        CIP_PRESS_MAX = 8.0  # bar
        
        # Pressions mixing (IM-10)
        MIX_PRESS_MIN = 2.0  # bar
        MIX_PRESS_MAX = 6.0  # bar

    class CorrelationLimits:
        """Limites validité des corrélations"""
        # Reynolds
        RE_MIN = 1          # Convection minimale
        RE_MIN_LAMINAR = 100    
        RE_MAX_LAMINAR = 2300   # Fin régime laminaire
        RE_TRANS_LOW = 2300     # Début transition
        RE_TRANS_HIGH = 4000    # Fin transition
        RE_MIN_TURB = 4000      # Début turbulent
        RE_MAX_TURB = 1e6       # Maximum corrélations
        
        # Autres nombres adimensionnels
        PR_MIN = 0.5        # Minimum Prandtl
        PR_MAX = 200.0      # Maximum Prandtl
        GR_MAX = 1e12       # Maximum Grashof
        NU_MIN = 0.1        # Minimum Nusselt
        
        # Stratification
        MAX_TEMP_GRADIENT = 2.5  # °C/m max
        MIN_STRAT_FACTOR = 0.2   # Mélange fort
        
        # Transfert thermique
        MIN_HEAT_TRANSFER = 1    # W/m².K
        MAX_HEAT_TRANSFER = 1e4  # W/m².K
        MIN_SURFACE = 1e-6       # m² (évite div/0)

    class PhysicalConstants:
        """Constantes physiques"""
        # Eau
        RHO_WATER = 1000.0     # kg/m³
        CP_WATER = 4186.0      # J/kg.K
        MU_WATER = 0.001       # Pa.s (à 20°C)
        K_WATER = 0.6          # W/m.K (à 20°C)
        BETA_WATER = 2.1e-4    # 1/K
        
        # Air
        RHO_AIR = 1.225       # kg/m³
        CP_AIR = 1006.0       # J/kg.K
        MU_AIR = 1.81e-5      # Pa.s
        K_AIR = 0.026         # W/m.K
        
        # Autres
        G = 9.81              # m/s²
        STEFAN_BOLTZMANN = 5.67e-8  # W/m².K⁴

    class SafetyFactors:
        """Facteurs de sécurité"""
        THERMAL = 1.1    # Calculs thermiques
        MECHANICAL = 1.2  # Calculs mécaniques
        
        # Facteurs spécifiques
        HEAT_TRANSFER = 1.15   # Coefficient d'échange
        PRESSURE_DROP = 1.2    # Pertes de charge
        MIXING_TIME = 1.25     # Temps de mélange

    class MaterialProperties:
        """Propriétés des matériaux"""
        # Conductivités thermiques (W/m.K)
        K_316L = 16.3
        K_MINERAL_WOOL = 0.04
        K_POLYURETHANE = 0.025
        
        # Résistances de contact (m².K/W)
        THERMAL_CONTACT = {
            '316L': {
                'mineral_wool': 0.0001,
                'polyurethane': 0.00015
            }
        }
        
        # Émissivités (-)
        EMISSIVITY = {
            '316L': 0.35,
            'mineral_wool': 0.9,
            'polyurethane': 0.85
        }

    class NumericalParams:
        """Paramètres numériques"""
        # Limites
        MIN_VALUE = 1e-10     # Évite division par zéro
        MAX_VALUE = 1e10      # Évite overflow
        MAX_ITERATIONS = 1000  # Boucles
        MAX_TIME = 7200       # Temps max simulation (s)
        
        # Tolérances
        TOL_TEMP = 0.1       # °C
        TOL_PRESS = 0.01     # bar
        TOL_CONVERGENCE = 1e-6  # Itérations
        
        # Pas de temps recommandés
        DT_HEATING = 30      # s
        DT_COOLING = 60      # s
        DT_MIXING = 10       # s

    @staticmethod
    def validate_operating_conditions(
        temp: float,
        pressure: float,
        mode: str = None
    ) -> None:
        """
        Valide les conditions opératoires
        Args:
            temp: température en °C
            pressure: pression en bar
            mode: None/'cip'/'mixing' pour limites spécifiques
        Raises:
            ValueError si hors limites
        """
        # Température
        if not Config.ProcessLimits.TEMP_MIN <= temp <= Config.ProcessLimits.TEMP_MAX:
            raise ValueError(
                f"Température {temp}°C hors limites "
                f"({Config.ProcessLimits.TEMP_MIN}-{Config.ProcessLimits.TEMP_MAX}°C)"
            )
            
        # Pression selon mode
        if mode == 'cip':
            if not Config.ProcessLimits.CIP_PRESS_MIN <= pressure <= Config.ProcessLimits.CIP_PRESS_MAX:
                raise ValueError(
                    f"Pression CIP {pressure}bar hors limites "
                    f"({Config.ProcessLimits.CIP_PRESS_MIN}-{Config.ProcessLimits.CIP_PRESS_MAX}bar)"
                )
        elif mode == 'mixing':
            if not Config.ProcessLimits.MIX_PRESS_MIN <= pressure <= Config.ProcessLimits.MIX_PRESS_MAX:
                raise ValueError(
                    f"Pression mixing {pressure}bar hors limites "
                    f"({Config.ProcessLimits.MIX_PRESS_MIN}-{Config.ProcessLimits.MIX_PRESS_MAX}bar)"
                )
        else:
            if not Config.ProcessLimits.PRESS_MIN <= pressure <= Config.ProcessLimits.PRESS_MAX:
                raise ValueError(
                    f"Pression {pressure}bar hors limites "
                    f"({Config.ProcessLimits.PRESS_MIN}-{Config.ProcessLimits.PRESS_MAX}bar)"
                )
            
    class PlateExchangerLimits:
        """Limites et paramètres standards échangeurs à plaques"""
        # Matériaux standards
        PLATE_MATERIAL_CONDUCTIVITY = {
            '316L': 16.3,  # W/m.K
            'AISI304': 14.9,  # W/m.K
        }

        # Géométrie typique
        PLATE_THICKNESS = 0.6     # mm
        PLATE_SPACING = 2.5       # mm
        CHEVRON_ANGLE = 30        # degrés
        
        # Facteurs de correction
        FOULING_WATER = 0.00017   # m².K/W (TEMA)
        FOULING_STEAM = 0.00009   # m².K/W (TEMA)
        SAFETY_FACTOR = 1.2       # Sur surface d'échange
        
        # Limites opératoires
        VELOCITY_MIN = 0.2        # m/s (évite encrassement)
        VELOCITY_MAX = 2.0        # m/s (érosion/vibrations)
        PRESSURE_DROP_MAX = 0.5   # bar
        
        # Performances standards
        HTC_MIN = 3000           # W/m².K (coefficient global min)
        LMTD_MIN = 5             # °C (pincement minimum)

    class ElectricalHeaterLimits:
        """Limites pour résistances électriques"""
        # Limites opératoires
        MIN_VOLTAGE = 200.0    # V
        MAX_VOLTAGE = 240.0    # V
        MIN_POWER = 0.5       # kW
        MAX_POWER = 12.0      # kW
        
        STRATIFICATION_FACTOR = 0.1  # Facteur de stratification
        
        # Limites thermiques
        MAX_SURFACE_TEMP = 750.0  # °C (Incoloy 800)
        MAX_POWER_DENSITY = 25.0  # kW/m²
        MIN_POWER_DENSITY = 5.0   # kW/m²
        
        # Facteurs de sécurité
        POWER_SAFETY = 0.9    # Facteur derating
        TEMP_SAFETY = 0.95    # Marge température

        # Matériaux standards
        MATERIALS = {
            'Incoloy800': {
                'max_temp': 750.0,  # °C
                'conductivity': 11.5  # W/m.K
            },
            '316L': {
                'max_temp': 550.0,
                'conductivity': 16.3
            }
        }