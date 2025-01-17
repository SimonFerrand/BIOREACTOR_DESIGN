# src/equipment/tanks/process_tank.py
from dataclasses import dataclass
from typing import Optional, Literal
from .bioreactor import TankGeometry, ThermalProperties

class ProcessTank:
    """Classe pour les tanks process (CIP, média, etc.)"""
    def __init__(
        self,
        geometry: TankGeometry,
        tank_type: Literal['cip', 'media', 'buffer'],
        material: str = "316L",
        insulation_type: Optional[str] = None,
        insulation_thickness: float = 0.0,
        design_pressure: float = 0.5,  # bar
        design_temperature: float = 95.0,  # °C
    ):
        self.geometry = geometry
        self.tank_type = tank_type
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
                'mineral_wool': 0.04,    # W/m.K
                'polyurethane': 0.025    # W/m.K
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

def create_standard_cip_tank(volume: float) -> ProcessTank:
    """Crée un tank CIP standard"""
    specs = {
        500: {
            'diameter': 800,
            'height_cylinder': 900,
            'height_total': 1200,
            'volume_cone': 50,
            'volume_total': 550
        },
        1000: {
            'diameter': 1000,
            'height_cylinder': 1200,
            'height_total': 1600,
            'volume_cone': 100,
            'volume_total': 1100
        }
    }
    
    if volume not in specs:
        raise ValueError(f"Volume {volume}L non disponible dans les standards")
        
    spec = specs[volume]
    geometry = TankGeometry(
        diameter=spec['diameter'],
        height_cylinder=spec['height_cylinder'],
        height_total=spec['height_total'],
        volume_useful=volume,
        volume_cone=spec['volume_cone'],
        volume_total=spec['volume_total']
    )
    
    return ProcessTank(
        geometry=geometry,
        tank_type='cip',
        insulation_type='mineral_wool',
        insulation_thickness=50.0
    )