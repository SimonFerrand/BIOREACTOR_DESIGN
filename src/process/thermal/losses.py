# src/process/thermal/losses.py
from typing import Optional, Dict, Union
import numpy as np
import logging
from .thermal_utils import ThermalProperties
from src.config import Config

# Configuration du logging
logger = logging.getLogger(__name__)

class ThermalCalculator:
    """
    Calculs des pertes thermiques avec modèle multi-zones
    Prend en compte :
    - Géométrie cylindro-conique
    - Stratification thermique
    - Isolation thermique
    - Convection naturelle et forcée
    """
    
    def __init__(self, tank):
        """
        Args:
            tank: instance de Bioreactor ou ProcessTank
        Raises:
            ValueError si configuration invalide
        """
        self.tank = tank
        self._validate_tank()
        
    def _validate_tank(self) -> None:
        """Valide la configuration du tank"""
        # Vérification dimensions physiques
        if self.tank.geometry.diameter <= 0:
            raise ValueError("Diamètre doit être positif")
            
        if self.tank.geometry.height_total <= self.tank.geometry.height_cylinder:
            raise ValueError(
                f"Hauteur totale ({self.tank.geometry.height_total}mm) doit être "
                f"supérieure à hauteur cylindrique ({self.tank.geometry.height_cylinder}mm)"
            )
            
        # Vérification matériaux
        if not hasattr(self.tank.wall, 'conductivity'):
            raise ValueError("Propriétés thermiques paroi manquantes")
            
        if self.tank.insulation:
            if not hasattr(self.tank.insulation, 'conductivity'):
                raise ValueError("Propriétés isolation manquantes")
            
            if self.tank.insulation.thickness <= 0:
                raise ValueError("Épaisseur isolation doit être positive")
                
        logger.info(
            f"Tank validé: {self.tank.geometry.volume_total}L, "
            f"isolation: {self.tank.insulation_type if self.tank.insulation else 'none'}"
        )
        
    def calculate_zone_dimensions(self, zone: str) -> Dict[str, float]:
        """
        Calcule les dimensions caractéristiques d'une zone
        Args:
            zone: 'top', 'cylinder', 'cone' ou 'bottom'
        Returns:
            dict avec dimensions en m
        """
        # Conversion mm -> m
        diameter = self.tank.geometry.diameter / 1000
        radius = diameter / 2
        height_cyl = self.tank.geometry.height_cylinder / 1000
        height_total = self.tank.geometry.height_total / 1000
        height_cone = height_total - height_cyl
        
        # Structure de retour
        dimensions = {
            'diameter': diameter,
            'radius': radius,
            'height': height_cyl if zone == 'cylinder' else height_cone,
            'volume': 0.0,
            'surface': 0.0,
            'char_length': 0.0
        }
        
        # Calculs selon zone
        if zone == 'cylinder':
            # Zone cylindrique
            volume = np.pi * radius**2 * height_cyl
            surface = np.pi * diameter * height_cyl
            char_length = height_cyl
            
            dimensions.update({
                'volume': volume,
                'surface': surface,
                'char_length': char_length
            })
            
        elif zone == 'cone':
            # Zone conique
            slant_height = np.sqrt(height_cone**2 + radius**2)
            lateral_surface = np.pi * radius * slant_height
            base_surface = np.pi * radius**2
            volume = np.pi * radius**2 * height_cone / 3
            
            dimensions.update({
                'volume': volume,
                'surface': lateral_surface + base_surface,
                'char_length': slant_height,
                'slant_height': slant_height,
                'lateral_surface': lateral_surface,
                'base_surface': base_surface
            })
            
        else:  # top ou bottom
            # Zones planes avec correction calotte
            surface = np.pi * radius**2
            volume = np.pi * radius**3 / 3  # Approximation calotte
            
            dimensions.update({
                'volume': volume,
                'surface': surface,
                'char_length': diameter
            })
            
        # Validation surfaces minimales
        for key in ['surface', 'char_length']:
            if dimensions[key] < Config.CorrelationLimits.MIN_SURFACE:
                logger.warning(
                    f"{key} très faible ({dimensions[key]:.2e}m²) "
                    f"pour zone {zone}"
                )
                dimensions[key] = Config.CorrelationLimits.MIN_SURFACE
                
        return dimensions
        
    def calculate_thermal_resistances(
        self,
        surface: float,      # m²
        h_int: float,        # W/m².K
        h_ext: float,        # W/m².K
        zone: str = ""       # pour logging
    ) -> Dict[str, float]:
        """
        Calcule les résistances thermiques de la paroi
        Returns:
            dict avec résistances en K/W
        """
        # Protection surface nulle
        surface = max(surface, Config.CorrelationLimits.MIN_SURFACE)
        
        # Résistance de contact selon matériaux
        R_contact = Config.MaterialProperties.THERMAL_CONTACT.get(
            self.tank.material, {}
        ).get(self.tank.insulation_type, 0.0001)  # m².K/W
        
        # 1. Convection interne
        R_conv_int = 1 / (h_int * surface)
        
        # 2. Conduction paroi
        R_wall = (self.tank.wall.thickness * 1e-3) / \
                (self.tank.wall.conductivity * surface)
        
        # 3. Isolation si présente
        if self.tank.insulation:
            R_ins = (self.tank.insulation.thickness * 1e-3) / \
                   (self.tank.insulation.conductivity * surface)
            R_total_cond = R_wall + R_contact * (1/surface) + R_ins
        else:
            R_ins = 0.0
            R_total_cond = R_wall
            
        # 4. Convection externe
        R_conv_ext = 1 / (h_ext * surface)
        
        # Résistance totale
        R_total = R_conv_int + R_total_cond + R_conv_ext
        
        # Validation des résultats
        min_resistance = Config.NumericalParams.MIN_VALUE
        if R_total < min_resistance:
            logger.warning(
                f"Résistance thermique très faible ({R_total:.2e}K/W) "
                f"pour zone {zone}"
            )
            R_total = min_resistance
            
        return {
            'R_conv_int': R_conv_int,
            'R_wall': R_wall,
            'R_contact': R_contact * (1/surface) if self.tank.insulation else 0,
            'R_insulation': R_ins,
            'R_conv_ext': R_conv_ext,
            'R_total': R_total,
            'surface': surface
        }
        
    def calculate_zone_heat_loss(
        self,
        zone: str,
        temp_fluid: float,
        temp_ambient: float,
        height: float = 0,
        pressure: float = 1.0,
        flow_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """Calcule les pertes thermiques pour une zone"""
        # Protection température avec marge
        temp_fluid_safe = ThermalProperties.validate_process_temperature(temp_fluid)
        
        # Dimensions de la zone
        dims = self.calculate_zone_dimensions(zone)
        
        # Température avec stratification
        if height > 0:
            # Gradient thermique variable selon Reynolds
            if flow_rate:
                # Vitesse caractéristique
                velocity = (flow_rate / 3600) / (np.pi * dims['radius']**2)
                re = ThermalProperties.calculate_reynolds(
                    velocity, dims['char_length'], temp_fluid_safe, pressure,
                    context=f"stratification {zone}"
                )
                # Réduction gradient si turbulent
                gradient_reduction = min(re/Config.CorrelationLimits.RE_MIN_TURB, 0.8)
                gradient = Config.CorrelationLimits.MAX_TEMP_GRADIENT * \
                        (1 - gradient_reduction)
            else:
                gradient = Config.CorrelationLimits.MAX_TEMP_GRADIENT
                
            temp_fluid_local = ThermalProperties.validate_process_temperature(
                temp_fluid_safe + (height * gradient)
            )
            strat_info = {
                'gradient': gradient,
                'temp_local': temp_fluid_local
            }
        else:
            temp_fluid_local = temp_fluid_safe
            strat_info = {'gradient': 0, 'temp_local': temp_fluid_safe}
            
        # 4. Calcul vitesse si recirculation
        if flow_rate:
            velocity = (flow_rate / 3600) / (np.pi * dims['radius']**2)
        else:
            velocity = 0
            
        # 5. Coefficients de convection
        geometry = 'vertical' if zone in ['cylinder', 'cone'] else 'horizontal'
        
        # Interne
        conv_int = ThermalProperties.calculate_convection_coefficient(
            temp_surf=temp_fluid_local - 2,  # Hypothèse ΔT paroi
            temp_fluid=temp_fluid_local,
            length=dims['char_length'],
            velocity=velocity,
            pressure=pressure,
            geometry=geometry
        )
        
        # Externe (air)
        conv_ext = ThermalProperties.calculate_convection_coefficient(
            temp_surf=temp_ambient + 2,
            temp_fluid=temp_ambient,
            length=dims['char_length'],
            velocity=0,  # air statique
            pressure=1.0,  # pression atmosphérique
            geometry=geometry
        )
        
        # 6. Résistances thermiques
        resistances = self.calculate_thermal_resistances(
            surface=dims['surface'],
            h_int=conv_int['h_total'],
            h_ext=conv_ext['h_total'],
            zone=zone
        )
        
        # 7. Pertes thermiques avec facteur sécurité
        delta_t = temp_fluid_local - temp_ambient
        heat_loss = (delta_t / resistances['R_total']) * \
                   Config.SafetyFactors.THERMAL
        
        return {
            'loss': max(0, heat_loss),
            'dimensions': dims,
            'stratification': strat_info,
            'convection_int': conv_int,
            'convection_ext': conv_ext,
            'resistances': resistances,
            'delta_t': delta_t
        }

    def calculate_heat_loss(
        self,
        temp_fluid: float,      # °C
        temp_ambient: float,    # °C
        pressure: float = 1.0,  # bar
        flow_rate: Optional[float] = None,  # m³/h
        detailed: bool = False
    ) -> Union[float, Dict[str, float]]:

        """
        Calcule les pertes thermiques totales avec stratification
        Args:
            detailed: si True, retourne détails par zone
        Returns:
            pertes en W ou dict avec détails
        """ 

        # Zones et leurs hauteurs relatives
        zones = [
            ('bottom', 0),
            ('cone', 0.2),
            ('cylinder', 0.5),
            ('top', 1.0)
        ]
        
        logger.info("\nDétail des pertes thermiques par zone:")
        logger.info("-" * 40)
        logger.info(f"Température fluide: {temp_fluid}°C")
        logger.info(f"Température ambiante: {temp_ambient}°C")
        logger.info(f"Débit: {flow_rate if flow_rate else 0} m³/h")
        
        # Calcul par zone
        total_loss = 0
        zone_results = {}
        
        for zone, rel_height in zones:
            # Hauteur absolue
            height = rel_height * self.tank.geometry.height_total / 1000
            
            # Calcul pertes
            results = self.calculate_zone_heat_loss(
                zone, temp_fluid, temp_ambient, height, pressure, flow_rate
            )
            
            total_loss += results['loss']
            zone_results[zone] = results
            
            # Logging détaillé pour chaque zone
            logger.info(f"\nZone: {zone}")
            logger.info(f"- Surface: {results['dimensions']['surface']:.2f} m²")
            logger.info(f"- Température locale: {results['stratification']['temp_local']:.1f}°C")
            logger.info(f"- h interne: {results['convection_int']['h_total']:.1f} W/m².K")
            logger.info(f"- h externe: {results['convection_ext']['h_total']:.1f} W/m².K")
            logger.info(f"- R totale: {results['resistances']['R_total']:.3f} K/W")
            logger.info(f"- Pertes: {results['loss']/1000:.2f} kW")

        logger.info(f"\nPertes totales: {total_loss/1000:.2f} kW")

        if detailed:
            return {
                'total_loss': total_loss,
                'zones': zone_results
            }
        else:
            return {'total_loss': total_loss}