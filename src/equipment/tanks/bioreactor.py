# src/equipment/tanks/bioreactor.py
# (ancien vessel.py)
from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class TankGeometry:                    # Renommé de VesselGeometry
    """Classe définissant la géométrie cylindro-conique"""
    diameter: float  # mm
    height_cylinder: float  # mm
    height_total: float  # mm
    volume_useful: float  # L
    volume_cone: float  # L
    volume_total: float  # L
    
    def __post_init__(self):
        if self.volume_total < self.volume_useful:
            raise ValueError("Le volume total doit être supérieur au volume utile")
        if self.height_total < self.height_cylinder:
            raise ValueError("La hauteur totale doit être supérieure à la hauteur cylindrique")
            
    @property
    def cone_angle(self) -> float:
        """Calcul de l'angle du cône en degrés"""
        height_cone = self.height_total - self.height_cylinder
        return np.degrees(np.arctan(self.diameter/(2*height_cone)))
    
    @property
    def surface_exchange(self) -> float:
        """Calcul de la surface d'échange thermique totale (m²)"""
        surface_cylinder = np.pi * self.diameter * self.height_cylinder
        height_cone = self.height_total - self.height_cylinder
        surface_cone = np.pi * self.diameter/2 * np.sqrt(height_cone**2 + (self.diameter/2)**2)
        return (surface_cylinder + surface_cone) / 1e6  # conversion en m²

@dataclass
class ThermalProperties:
    """Propriétés thermiques des matériaux"""
    conductivity: float  # W/m.K
    thickness: float  # mm

class Bioreactor:                     # Renommé de Vessel
    """Classe principale pour le bioréacteur"""
    def __init__(
        self,
        geometry: TankGeometry,       # Mise à jour du type
        material: str = "316L",
        insulation_type: Optional[str] = None,
        insulation_thickness: float = 0.0,
        design_pressure: float = 0.5,
        design_temperature: float = 95.0,
    ):
        self.geometry = geometry
        self.material = material
        self.insulation_type = insulation_type
        self.insulation_thickness = insulation_thickness
        self.design_pressure = design_pressure
        self.design_temperature = design_temperature
        
        self.wall = ThermalProperties(
            conductivity=16.3,  # W/m.K (316L)
            thickness=self.calculate_wall_thickness()
        )
        
        if insulation_type:
            conductivity = {
                'mineral_wool': 0.04,
                'polyurethane': 0.025
            }[insulation_type]
            self.insulation = ThermalProperties(
                conductivity=conductivity,
                thickness=insulation_thickness
            )
        else:
            self.insulation = None

    def calculate_wall_thickness(self) -> float:
        """Calcule l'épaisseur minimale requise selon la norme EN 13445"""
        allowable_stress = 205  # MPa à 95°C
        weld_efficiency = 0.85
        corrosion_allowance = 1.0  # mm
        
        pressure = self.design_pressure * 1e5  # conversion en Pa
        diameter = self.geometry.diameter
        thickness = (pressure * diameter) / (2 * allowable_stress * weld_efficiency * 1e6)
        return thickness * 1000 + corrosion_allowance

def create_standard_bioreactor(volume: float) -> Bioreactor:  # Renommé de create_standard_vessel
    """Crée une instance de bioréacteur standard selon les données du fabricant"""
    specs = {
        2000: {
            'diameter': 1310,
            'height_cylinder': 1500,
            'height_total': 3290,
            'volume_cone': 510,
            'volume_total': 2510
        }
    }
    
    if volume not in specs:
        raise ValueError(f"Volume {volume}L non disponible dans les standards")
        
    spec = specs[volume]
    geometry = TankGeometry(          # Mise à jour du type
        diameter=spec['diameter'],
        height_cylinder=spec['height_cylinder'],
        height_total=spec['height_total'],
        volume_useful=volume,
        volume_cone=spec['volume_cone'],
        volume_total=spec['volume_total']
    )
    
    return Bioreactor(geometry=geometry)