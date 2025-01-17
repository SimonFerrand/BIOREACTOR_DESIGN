# src/equipment/heat_system/plate_exchanger.py
from dataclasses import dataclass
from typing import Dict, Optional
from src.process.thermal.thermal_utils import ThermalProperties
from src.process.thermal.thermal_calculations import ThermalCalculations
from src.config import Config

@dataclass
class PlateExchangerSpecs:
    """Spécifications de l'échangeur à plaques"""
    surface: float  # m²
    nb_plates: int
    power: float  # kW
    grouping: str  # ex: "2x3/2x3"
    material: str = '316L'   # Matériau par défaut
    height: float = 0.5  # m (hauteur standard plaques)
    
    @property
    def plates_per_pass(self) -> tuple:
        """Retourne le nombre de plaques par passe (primaire, secondaire)"""
        p1, p2 = self.grouping.split('/')
        return (int(p1.split('x')[0]), int(p2.split('x')[0]))

class PlateHeatExchanger:
    """Classe pour l'échangeur à plaques"""
    def __init__(
        self,
        surface: float,
        nb_plates: int,
        power: float,
        grouping: str,
        design_pressure: float = 10.0,  # bar
        max_temperature: float = 150.0  # °C
    ):
        self.specs = PlateExchangerSpecs(
            surface=surface,
            nb_plates=nb_plates,
            power=power,
            grouping=grouping
        )
        self.design_pressure = design_pressure
        self.max_temperature = max_temperature
        
    def calculate_heat_transfer(
        self,
        flow_rate_primary: float,    # m³/h
        temp_in_primary: float,      # °C 
        temp_out_primary: float,     # °C
        flow_rate_secondary: float,  # m³/h
        temp_in_secondary: float,    # °C
    ) -> tuple[float, float]:
        """
        Calcule le transfert thermique dans l'échangeur selon standards TEMA
        Args:
            flow_rate_primary: Débit primaire (vapeur)
            temp_in_primary: T entrée primaire
            temp_out_primary: T sortie primaire attendue 
            flow_rate_secondary: Débit secondaire (eau)
            temp_in_secondary: T entrée secondaire
        Returns:
            puissance échangée (kW), T sortie secondaire (°C)
        """
        # 1. Géométrie des canaux
        nb_channels_primary, nb_channels_secondary = self.specs.plates_per_pass
        plate_spacing = Config.PlateExchangerLimits.PLATE_SPACING / 1000  # m
        dh = 2 * plate_spacing  # diamètre hydraulique
        width = (self.specs.surface / self.specs.nb_plates / 
                Config.PlateExchangerLimits.PLATE_HEIGHT)  # m
        flow_area = width * plate_spacing  # m²
        
        # 2. Propriétés physiques moyennes
        props_steam = ThermalProperties.get_steam_properties(self.design_pressure)
        props_cold = ThermalProperties.get_water_properties(
            temp_in_secondary, self.design_pressure, context="exchanger"
        )
        
        # 3. Débits massiques par canal
        mass_flow_steam = flow_rate_primary * props_steam['rho'] / (3600 * nb_channels_primary)  # kg/s
        mass_flow_water = flow_rate_secondary * props_cold['rho'] / (3600 * nb_channels_secondary)  # kg/s
        
        # 4. Coefficient vapeur (condensation)
        cond_results = ThermalCalculations.calculate_condensation_film(
            temp_steam=temp_in_primary,
            temp_wall=temp_in_secondary + 5,  # Estimation
            pressure=self.design_pressure,
            height=Config.PlateExchangerLimits.PLATE_HEIGHT
        )
        h_steam = cond_results['h']
        
        # 5. Coefficient eau (convection plaques)
        water_results = ThermalCalculations.calculate_plate_heat_transfer(
            mass_flow=mass_flow_water,
            temp=temp_in_secondary,
            pressure=self.design_pressure,
            dh=dh,
            area=flow_area
        )
        h_water = water_results['h']
        
        # 6. Coefficient global avec facteurs d'encrassement
        k_plate = Config.PlateExchangerLimits.PLATE_MATERIAL_CONDUCTIVITY[self.specs.material]
        U = 1 / (
            1/h_steam + Config.PlateExchangerLimits.FOULING_STEAM +
            Config.PlateExchangerLimits.PLATE_THICKNESS*1e-3/k_plate +
            1/h_water + Config.PlateExchangerLimits.FOULING_WATER
        )
        
        # 7. Estimation température sortie
        q_max = mass_flow_water * props_cold['cp'] * (temp_in_primary - temp_in_secondary)
        dtlm = ThermalCalculations.calculate_lmtd(
            t_hot_in=temp_in_primary,
            t_hot_out=temp_out_primary,
            t_cold_in=temp_in_secondary,
            t_cold_out=temp_in_secondary + 20  # Première estimation
        )
        
        # 8. Puissance échangée avec facteur sécurité
        q = U * self.specs.surface * dtlm * Config.PlateExchangerLimits.SAFETY_FACTOR
        q = min(q, q_max)  # Limitation physique
        
        # 9. Température sortie réelle
        temp_out_secondary = temp_in_secondary + q/(mass_flow_water * nb_channels_secondary * props_cold['cp'])
        
        return q/1000, temp_out_secondary  # kW, °C


    def calculate_pressure_drop(
        self,
        flow_rate: float,  # m³/h
        temp: float,       # °C
        is_primary: bool = False  # True = vapeur, False = eau
    ) -> float:
        """
        Calcule perte de charge dans l'échangeur à plaques
        Args:
            flow_rate: débit volumique
            temp: température fluide
            is_primary: True si circuit vapeur
        Returns:
            Perte de charge (bar)
        """
        # Géométrie spécifique à cet échangeur
        nb_channels = self.specs.plates_per_pass[0 if is_primary else 1]
        plate_spacing = Config.PlateExchangerLimits.PLATE_SPACING / 1000  # m
        
        # Propriétés fluide selon circuit
        props = (ThermalProperties.get_steam_properties(self.design_pressure) if is_primary 
                else ThermalProperties.get_water_properties(temp, self.design_pressure))
        
        # Débit par canal
        mass_flow = flow_rate * props['rho'] / (3600 * nb_channels)
        velocity = mass_flow/(props['rho'] * plate_spacing**2)
        
        # Reynolds
        re = props['rho'] * velocity * (2*plate_spacing) / props['mu']
        
        # Facteur friction selon régime
        if re < Config.CorrelationLimits.RE_MAX_LAMINAR:
            f = 16/re
        else:
            f = 0.72 * re**(-0.25) * (Config.PlateExchangerLimits.CHEVRON_ANGLE/30)**0.5
            
        # Pertes singulières (entrée/sortie)
        k_sing = 1.5
        
        # Pertes totales
        dp = (f * self.specs.height/plate_spacing + k_sing) * props['rho'] * velocity**2 / 2
        
        return dp/1e5  # Pa -> bar


    def calculate_plate_heat_transfer(
        self,                  # Ajout de self car c'est une méthode d'instance
        flow_rate: float,      # m³/h (au lieu de mass_flow)
        temp: float,          # °C
        pressure: float,      # bar
        dh: float = None,     # m (optionnel)
        area: float = None    # m² (optionnel)
    ) -> Dict[str, float]:
        """
        Calcule coefficient convection plaques (corrélation Martin)
        """
        # Utiliser les valeurs par défaut si non spécifiées
        if dh is None:
            dh = Config.PlateExchangerLimits.PLATE_SPACING / 1000  # m
        if area is None:
            area = self.specs.surface / self.specs.nb_plates  # m²
        
        # Propriétés à température moyenne
        props = ThermalProperties.get_water_properties(
            temp, pressure, context="exchanger"
        )
        
        # Conversion débit volumique en débit massique
        mass_flow = flow_rate * props['rho'] / 3600  # kg/s
        
        # Nombres adimensionnels
        velocity = mass_flow / (props['rho'] * area)
        re = props['rho'] * velocity * dh / props['mu']
        pr = props['cp'] * props['mu'] / props['k']
        
        # Nusselt Martin PHE
        if re < Config.CorrelationLimits.RE_MAX_LAMINAR:
            nu = 0.374 * (Config.PlateExchangerLimits.CHEVRON_ANGLE/30)**0.13 * re**0.5 * pr**(1/3)
        else:
            nu = 0.122 * (Config.PlateExchangerLimits.CHEVRON_ANGLE/30)**0.374 * re**0.7 * pr**(1/3)
            
        # Coefficient convection
        h = nu * props['k'] / dh
        
        return {
            'h': h,
            're': re,
            'pr': pr,
            'velocity': velocity,
            'properties': props
        }

def create_standard_exchanger(model: str) -> PlateHeatExchanger:
    """
    Crée un échangeur standard selon la documentation Barriquand
    Args:
        model: '3.5HL', '6HL' ou '11HL'
    """
    specs = {
        '3.5HL': {'surface': 0.89, 'power': 32, 'grouping': '2x3/2x3'},
        '6HL': {'surface': 1.54, 'power': 56, 'grouping': '2x5/2x5'},
        '11HL': {'surface': 2.51, 'power': 102, 'grouping': '2x8/2x8'}
    }
    
    if model not in specs:
        raise ValueError(f"Modèle {model} non disponible")
        
    spec = specs[model]
    nb_plates = sum(PlateExchangerSpecs(**spec, nb_plates=0).plates_per_pass)
    
    return PlateHeatExchanger(
        surface=spec['surface'],
        nb_plates=nb_plates,
        power=spec['power'],
        grouping=spec['grouping']
    )