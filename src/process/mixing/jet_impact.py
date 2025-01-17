# src/process/mixing/jet_impact.py
import numpy as np
from typing import Dict, Tuple
from src.equipment.spray_heads.rotary_jet_mixer import RotaryJetMixer

class JetFlowCalculator:
    """Calculs des débits et vitesses pour le jet rotatif"""
    def __init__(self, mixer: RotaryJetMixer):
        self.mixer = mixer
        
    def calculate_jet_velocity(self, pressure: float) -> float:
        """
        Calcule la vitesse du jet en sortie de buse
        Args:
            pressure: pression en bar
        Returns:
            vitesse en m/s
        """
        flow_rate = self.mixer.calculate_flow_rate(pressure)
        area = np.pi * (self.mixer.nozzle_diameter/1000)**2 / 4
        return (flow_rate / 3600) / area
        
    def calculate_reynolds_impact(
        self,
        pressure: float,
        distance: float,
        temperature: float = 20.0
    ) -> float:
        """
        Calcule le nombre de Reynolds à l'impact du jet
        Args:
            pressure: pression en bar
            distance: distance en m
            temperature: température en °C
        Returns:
            nombre de Reynolds
        """
        velocity = self.calculate_jet_velocity(pressure)
        diameter = self.mixer.nozzle_diameter / 1000  # conversion en m
        
        # Viscosité cinématique de l'eau en fonction de la température
        # Approximation polynomiale
        nu = (1.0 + (temperature - 20) * -0.02) * 1e-6  # m²/s
        
        return (velocity * diameter) / nu

class ShearStressCalculator:
    """Calculs des contraintes de cisaillement à l'impact"""
    @staticmethod
    def calculate_wall_shear_stress(
        velocity: float,
        distance: float,
        nozzle_diameter: float,
        fluid_density: float = 1000  # kg/m³
    ) -> float:
        """
        Calcule la contrainte de cisaillement à la paroi
        Args:
            velocity: vitesse du jet (m/s)
            distance: distance à la buse (m)
            nozzle_diameter: diamètre de la buse (mm)
            fluid_density: masse volumique du fluide (kg/m³)
        Returns:
            contrainte en Pa
        """
        # Modèle de jet impactant
        d = nozzle_diameter / 1000  # conversion en m
        cf = 0.0447  # coefficient de frottement (empirique)
        
        return 0.5 * fluid_density * velocity**2 * cf * (d/distance)**2

    @staticmethod
    def is_safe_for_algae(shear_stress: float) -> bool:
        """
        Vérifie si la contrainte est acceptable pour les microalgues
        Args:
            shear_stress: contrainte en Pa
        Returns:
            True si acceptable
        """
        # Valeurs typiques de résistance au cisaillement pour Chlorella
        MAX_SAFE_STRESS = 0.3  # Pa
        return shear_stress <= MAX_SAFE_STRESS