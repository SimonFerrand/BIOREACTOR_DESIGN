# src/equipment/spray_devices/rotary_jet_mixer.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class NozzleType(Enum):
    """Types de buses disponibles pour l'IM-10"""
    A = 5.5  # mm
    B = 4.6  # mm
    C = 3.9  # mm

@dataclass
class MixerSpecs:
    """Spécifications du mélangeur rotatif Alfa Laval IM-10"""
    # Pressions de travail selon documentation
    working_pressure_min: float = 2.0  # bar
    working_pressure_max: float = 8.0  # bar
    mixing_pressure_min: float = 2.0  # bar
    mixing_pressure_max: float = 6.0  # bar
    cip_pressure_min: float = 4.0  # bar
    cip_pressure_max: float = 8.0  # bar
    
    # Autres spécifications
    max_temperature: float = 95.0  # °C
    connection_size: float = 25.4  # mm (1")
    weight: float = 5.1  # kg

class RotaryJetMixer:
    """Classe pour le mélangeur rotatif Alfa Laval IM-10"""
    def __init__(
        self,
        nozzle_type: NozzleType = NozzleType.C,
        specs: Optional[MixerSpecs] = None
    ):
        self.nozzle_type = nozzle_type
        self.specs = specs or MixerSpecs()
        
        # Identifiants pour les courbes de débit (à utiliser dans mechanical_stress.py)
        self._flow_coeffs = {
            NozzleType.A: (1.8, 0.8),  # (pente, offset) - données fabricant
            NozzleType.B: (1.4, 0.6),
            NozzleType.C: (1.0, 0.4)
        }

    @property
    def nozzle_diameter(self) -> float:
        """Diamètre de la buse en mm"""
        return self.nozzle_type.value

    def validate_pressure(self, pressure: float, mode: str = 'mixing') -> None:
        """
        Vérifie si la pression est dans les limites autorisées
        Args:
            pressure: pression en bar
            mode: 'mixing' ou 'cip'
        """
        if mode == 'mixing':
            min_p = self.specs.mixing_pressure_min
            max_p = self.specs.mixing_pressure_max
        else:  # mode CIP
            min_p = self.specs.cip_pressure_min
            max_p = self.specs.cip_pressure_max
            
        if pressure < min_p or pressure > max_p:
            raise ValueError(
                f"Pression {pressure} bar hors limites pour mode {mode} "
                f"({min_p}-{max_p} bar)"
            )

    def calculate_flow_rate(self, pressure: float, mode: str = 'mixing') -> float:
        """
        Calcule le débit pour une pression donnée
        Args:
            pressure: pression en bar
            mode: 'mixing' ou 'cip'
        Returns:
            débit en m³/h
        """
        # Validation des pressions selon mode
        self.validate_pressure(pressure, mode)
            
        # Calcul débit selon courbes caractéristiques
        a, b = self._flow_coeffs[self.nozzle_type]
        return a * pressure + b