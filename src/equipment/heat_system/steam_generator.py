# src/equipment/heat_system/steam_generator.py
from typing import Dict, Optional, Any, List  
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class SteamGeneratorSpecs:
    """Spécifications du générateur de vapeur"""
    power: float  # kW
    steam_volume: float  # kg/h
    pressure: float  # bar
    model: str
    
class SteamGenerator:
    """Classe pour le générateur de vapeur"""
    def __init__(
        self,
        model: str,
        power: float,
        steam_volume: float,
        pressure: float = 2.5  # bar
    ):
        self.specs = SteamGeneratorSpecs(
            model=model,
            power=power,
            steam_volume=steam_volume,
            pressure=pressure
        )
        
    def calculate_steam_flow(
        self,
        power_needed: float  # kW
    ) -> float:
        """
        Calcule le débit de vapeur nécessaire
        Args:
            power_needed: puissance requise en kW
        Returns:
            débit de vapeur en kg/h
        """
        # Propriétés de la vapeur à 2.5 bar
        h_vap = 2164.4  # kJ/kg (enthalpie de vaporisation)
        
        # Conversion puissance kW -> kJ/h
        power_kj = power_needed * 3600
        
        # Débit de vapeur nécessaire
        steam_flow = power_kj / h_vap
        
        return min(steam_flow, self.specs.steam_volume)  # limité par la capacité
    
    def calculate_heating_power(
        self,
        steam_flow: float,  # kg/h
        condensate_return: bool = True
    ) -> float:
        """
        Calcule la puissance de chauffe disponible
        Args:
            steam_flow: débit de vapeur en kg/h
            condensate_return: si True, prise en compte du retour condensats
        Returns:
            puissance en kW
        """
        # Propriétés de la vapeur
        h_vap = 2164.4  # kJ/kg 
        h_cond = 504.7  # kJ/kg (enthalpie du condensat à 2.5 bar)
        h_feed = 37.8   # kJ/kg (enthalpie eau alimentation à 9°C)
        
        # Puissance totale
        if condensate_return:
            power_kj = steam_flow * (h_vap + h_cond - h_feed)
        else:
            power_kj = steam_flow * (h_vap - h_feed)
            
        return power_kj / 3600  # conversion en kW
    
    def calculate_steam_power_with_control(
        self,
        temp_reactor_inlet: float,  # °C (température avant entrée réacteur)
        temp_target: float,         # °C
        temp_return: float,         # °C (température retour tank)
        margin: float = 2.0,        # °C
        detailed: bool = False
    ) -> Dict[str, float]:
        """
        Calcule l'ouverture de vanne vapeur pour contrôle température
        Returns:
            dict avec pourcentage ouverture vanne et détails
        """
        # Protection température
        if temp_return >= self.specs.max_temperature:
            return {'valve_opening': 0, 'steam_flow': 0} if detailed else {'valve_opening': 0}
            
        # Erreur sur point de contrôle principal
        error = temp_target - temp_reactor_inlet
        
        # Calcul ouverture vanne
        if abs(error) <= margin:
            # Zone de contrôle fin
            opening = (error / margin) * 0.5  # Max 50% ouvert près de la cible
        else:
            # Pleine ouverture si loin
            opening = 1.0 if error > 0 else 0.0
            
        # Calcul débit vapeur résultant
        steam_flow = self.calculate_steam_flow(
            opening * self.specs.steam_volume  # kg/h
        )
        
        if detailed:
            return {
                'valve_opening': opening,
                'steam_flow': steam_flow,
                'error': error,
                'temp_measured': temp_reactor_inlet,
                'temp_return': temp_return
            }
        return {'valve_opening': opening}

def create_standard_generator(model: str) -> SteamGenerator:
    """
    Crée un générateur standard selon la documentation
    Args:
        model: 'TD9', 'TD13', 'TD16', 'TD23' ou 'TD33'
    """
    specs = {
        'TD9': {'power': 9.9, 'steam': 15},
        'TD13': {'power': 13.2, 'steam': 20},
        'TD16': {'power': 16.5, 'steam': 25},
        'TD23': {'power': 23.1, 'steam': 35},
        'TD33': {'power': 33.0, 'steam': 50}
    }
    
    if model not in specs:
        raise ValueError(f"Modèle {model} non disponible")
        
    spec = specs[model]
    return SteamGenerator(
        model=model,
        power=spec['power'],
        steam_volume=spec['steam']
    )