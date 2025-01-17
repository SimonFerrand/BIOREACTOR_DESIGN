# src/equipment/piping/pipeline.py
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from src.config import Config
from src.process.thermal.thermal_utils import ThermalProperties

@dataclass
class PipeSpecs:
    """Spécifications des tuyaux"""
    diameter: float      # mm (diamètre interne)
    length: float       # m
    thickness: float    # mm (épaisseur paroi)
    insulation_type: Optional[str] = None
    insulation_thickness: float = 0.0  # mm
    
    def __post_init__(self):
        """Validation des paramètres"""
        if self.diameter <= 0:
            raise ValueError(f"Diamètre invalide: {self.diameter}mm")
        if self.length <= 0:
            raise ValueError(f"Longueur invalide: {self.length}m")
        if self.thickness <= 0:
            raise ValueError(f"Épaisseur invalide: {self.thickness}mm")
        if self.insulation_thickness < 0:
            raise ValueError("Épaisseur isolation invalide")

class Pipeline:
    """Modélisation des tuyaux avec pertes thermiques"""
    
    def __init__(
        self,
        specs: PipeSpecs,
        material: str = "316L",
        design_pressure: float = 10.0,  # bar
        design_temperature: float = 95.0  # °C
    ):
        self.specs = specs
        self.material = material
        self.design_pressure = design_pressure
        self.design_temperature = design_temperature
        
        # Validation design
        Config.validate_operating_conditions(
            design_temperature, 
            design_pressure, 
            mode='cip'
        )
        
        # Volume interne
        self._calculate_volume()
        
        # Propriétés thermiques
        self._setup_thermal_properties()
        
    def _calculate_volume(self):
        """Calcule volume interne"""
        self.volume = (
            np.pi * (self.specs.diameter/1000/2)**2 * 
            self.specs.length * 1000  # L
        )
        
    def _setup_thermal_properties(self):
        """Configure propriétés thermiques"""
        # Paroi
        self.wall_conductivity = (
            Config.MaterialProperties.K_316L 
            if self.material == "316L"
            else 15.0  # W/m.K approximation autres aciers
        )
        
        # Isolation si présente
        if self.specs.insulation_type:
            self.insulation_conductivity = {
                'mineral_wool': Config.MaterialProperties.K_MINERAL_WOOL,
                'polyurethane': Config.MaterialProperties.K_POLYURETHANE
            }[self.specs.insulation_type]
        else:
            self.insulation_conductivity = None

    def calculate_heat_loss(
        self,
        temp_fluid: float,    # °C
        temp_ambient: float,  # °C 
        flow_rate: float,     # m³/h
        pressure: float = 1.0 # bar
    ) -> Dict[str, float]:
        """
        Calcule pertes thermiques avec modèle détaillé
        Returns:
            dict avec pertes (W) et paramètres
        """
        # 1. Validation températures
        Config.validate_operating_conditions(temp_fluid, pressure)
        Config.validate_operating_conditions(temp_ambient, pressure)
        
        # 2. Surface d'échange
        surface = np.pi * self.specs.diameter/1000 * self.specs.length  # m²
        
        # 3. Coefficients d'échange
        h_int = self._calculate_internal_htc(temp_fluid, flow_rate, pressure)
        h_ext = self._calculate_external_htc(temp_ambient)
        
        # 4. Résistances thermiques
        R_conv_int = 1 / (h_int * surface)
        
        R_wall = np.log((self.specs.diameter + 2*self.specs.thickness) / 
                       self.specs.diameter) / \
                (2 * np.pi * self.wall_conductivity * self.specs.length)
        
        if self.insulation_conductivity:
            d_ext = (self.specs.diameter + 2*self.specs.thickness + 
                    2*self.specs.insulation_thickness)
            R_ins = np.log(d_ext/(self.specs.diameter + 2*self.specs.thickness)) / \
                    (2 * np.pi * self.insulation_conductivity * self.specs.length)
            R_conv_ext = 1 / (h_ext * np.pi * d_ext/1000 * self.specs.length)
        else:
            R_ins = 0
            d_ext = self.specs.diameter + 2*self.specs.thickness
            R_conv_ext = 1 / (h_ext * np.pi * d_ext/1000 * self.specs.length)
            
        R_total = R_conv_int + R_wall + R_ins + R_conv_ext
        
        # 5. Pertes thermiques avec facteur sécurité
        q_loss = (temp_fluid - temp_ambient) / R_total
        q_loss *= Config.SafetyFactors.THERMAL
        
        return {
            'heat_loss': q_loss,
            'h_internal': h_int,
            'h_external': h_ext,
            'R_total': R_total,
            'surface': surface
        }
        
    def _calculate_internal_htc(
        self,
        temp: float,      # °C
        flow_rate: float, # m³/h
        pressure: float = 1.0  # bar
    ) -> float:
        """Calcule coefficient convection interne"""
        # Vitesse moyenne
        area = np.pi * (self.specs.diameter/1000/2)**2
        velocity = (flow_rate/3600) / area  # m/s
        
        # Propriétés fluide
        props = ThermalProperties.get_water_properties(
            temp, pressure, context="process"
        )
        
        # Reynolds
        re = ThermalProperties.calculate_reynolds(
            velocity=velocity,
            length=self.specs.diameter/1000,
            temp=temp,
            pressure=pressure,
            context="pipe"
        )
        
        # Nusselt selon régime
        if re < Config.CorrelationLimits.RE_MAX_LAMINAR:
            # Laminaire établi
            nu = 3.66
        else:
            # Turbulent (Dittus-Boelter)
            nu = 0.023 * re**0.8 * props['Pr']**0.4
            
        # Coefficient avec sécurité
        h = nu * props['k'] / (self.specs.diameter/1000)
        h *= Config.SafetyFactors.HEAT_TRANSFER
        
        return h
    
    def _calculate_external_htc(
        self,
        temp_ambient: float,  # °C
    ) -> float:
        """Calcule coefficient convection externe"""
        return 10.0  # W/m².K (approximation air libre)