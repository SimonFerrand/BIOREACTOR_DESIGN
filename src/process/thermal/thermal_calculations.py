# src/process/thermal/thermal_calculations.py
from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple
import math
import logging
from src.process.thermal.thermal_utils import ThermalProperties
from src.config import Config

@dataclass
class ConvectionCoefficients:
    """Coefficients d'échange standards (W/m².K)"""
    # Valeurs TEMA / VDI Heat Atlas
    WATER_MIN: float = 1500  # Eau liquide min
    WATER_MAX: float = 10000  # Eau liquide max
    STEAM_CONDENSING_MIN: float = 4000  # Vapeur condensante min
    STEAM_CONDENSING_MAX: float = 12000  # Vapeur condensante max
    
class ThermalCalculations:
    """
    Calculs thermiques standards échangeur eau/vapeur selon TEMA
    """
    @staticmethod
    def calculate_lmtd(
        t_hot_in: float,
        t_hot_out: float,
        t_cold_in: float, 
        t_cold_out: float,
        flow_type: str = "counter"
    ) -> float:
        """
        Calcule la différence de température logarithmique moyenne
        Args:
            flow_type: "counter" ou "parallel"
        """
        dt1 = t_hot_in - t_cold_out
        dt2 = t_hot_out - t_cold_in
        
        if abs(dt1 - dt2) < 0.1:  # Évite division par zéro
            return dt1
            
        lmtd = (dt1 - dt2) / np.log(dt1/dt2)
        
        # Facteur correctif selon configuration
        if flow_type == "parallel":
            F = 0.9  # Approx pour échangeur à plaques
        else:
            F = 1.0
            
        return lmtd * F
        
    @staticmethod
    def calculate_condensation_coeff(
        temp: float,      # °C
        pressure: float,  # bar
        height: float,    # m
        incondensables: float = 0.001  # Fraction massique
    ) -> float:
        """
        Calcule le coefficient de condensation en film (Nusselt)
        Inclut correction pour incondensables
        """
        # Propriétés physiques vapeur saturée
        rho_l = 958  # kg/m³ à 100°C 
        rho_v = 0.6  # kg/m³ à 2.5 bar
        mu_l = 0.282E-3  # Pa.s
        k_l = 0.68  # W/m.K
        g = 9.81  # m/s²
        
        # Nusselt condensation en film
        h_cond = 0.943 * (
            (rho_l * (rho_l - rho_v) * g * k_l**3 * height) /
            (mu_l * (temp - pressure*100))
        )**(1/4)
        
        # Correction pour incondensables (Rose)
        f_nc = 1 / (1 + 1.3*incondensables)
        
        h_cond *= f_nc
        
        # Limites physiques
        return min(
            max(h_cond, ConvectionCoefficients.STEAM_CONDENSING_MIN),
            ConvectionCoefficients.STEAM_CONDENSING_MAX
        )
        
    @staticmethod 
    def calculate_water_coeff(
        flow_rate: float,  # m³/h
        temp: float,      # °C
        dh: float,        # m (diamètre hydraulique)
        roughness: float = 0.015  # mm
    ) -> float:
        """
        Calcule le coefficient convection eau
        Inclut correction pour rugosité
        """
        # Propriétés eau fonction température
        rho = 1000 - 0.11*(temp - 20)  # kg/m³
        mu = (2.414E-5)*10**(247.8/(temp + 133.15))  # Pa.s
        k = 0.58 + 0.00167*(temp - 20)  # W/m.K
        cp = 4186  # J/kg.K
        
        # Vitesse et Reynolds
        area = math.pi * dh**2 / 4
        velocity = (flow_rate/3600) / area
        re = rho * velocity * dh / mu
        pr = mu * cp / k
        
        # Facteur friction (Haaland)
        f = (-1.8 * np.log10(
            (roughness/1000/dh/3.7)**1.11 + 
            6.9/re
        ))**(-2)
        
        # Nusselt (Gnielinski)
        nu = (f/8 * (re-1000) * pr) / \
             (1 + 12.7 * (f/8)**0.5 * (pr**(2/3) - 1))
             
        h_conv = nu * k / dh
        
        # Limites physiques
        return min(
            max(h_conv, ConvectionCoefficients.WATER_MIN),
            ConvectionCoefficients.WATER_MAX
        )

    @staticmethod
    def calculate_condensation_film(
        temp_steam: float,  # °C
        temp_wall: float,   # °C
        pressure: float,    # bar
        height: float,      # m
        inclination: float = 90  # degrés (vertical)
    ) -> Dict[str, float]:
        """
        Calcule coefficient condensation en film (Nusselt)
        """
        # Validation des entrées
        if np.isnan(temp_steam) or np.isnan(temp_wall) or np.isnan(pressure) or np.isnan(height):
            raise ValueError(f"Paramètres invalides: temp_steam={temp_steam}, temp_wall={temp_wall}, pressure={pressure}, height={height}")

        # Protection différence température
        delta_t = max(temp_steam - temp_wall, 0.1)  # Au moins 0.1°C de différence

        # Propriétés condensat à température film
        t_film = (temp_steam + temp_wall) / 2
        props = ThermalProperties.get_water_properties(
            t_film, pressure, context="exchanger"
        )
        
        # Propriétés vapeur
        props_steam = ThermalProperties.get_steam_properties(pressure)
        
        # Vérification propriétés valides
        if not all(props.values()) or not all(props_steam.values()):
            raise ValueError("Propriétés thermiques invalides")

        # Facteurs physiques
        g = Config.PhysicalConstants.G
        h_fg = props_steam['h_vaporization'] * 1000  # J/kg
        
        # Protection densités
        rho_diff = max(props['rho'] - props_steam['rho'], 0.1)
        
        # Facteur inclinaison
        f_angle = np.sin(np.radians(inclination))
        f_angle = max(f_angle, 0.01)  # Évite division par zéro
        
        try:
            # Protection contre les différences de densité négatives/nulles
            rho_diff = props['rho'] - props_steam['rho']
            if rho_diff <= 0:
                rho_diff = 0.1  # Valeur minimale pour éviter NaN
                
            term = (
                props['rho'] * rho_diff * \
                g * props['k']**3 * h_fg * f_angle / \
                (props['mu'] * delta_t * height)
            )
            
            if term <= 0:
                raise ValueError("Terme de calcul négatif ou nul")
                
            h_cond = 0.943 * term**(1/4)
            
            # Vérification résultat
            if np.isnan(h_cond) or h_cond <= 0:
                raise ValueError(f"Coefficient invalide: {h_cond}")
                
            return {
                'h': h_cond,
                'properties': props,
                'steam': props_steam,
                'delta_t': delta_t,
                'term': term
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul condensation: {str(e)}")
            logger.debug(f"Paramètres: term={term}, props={props}, steam={props_steam}")
            raise ValueError("Échec calcul coefficient condensation") from e
    

class CirculationHeatTransfer:
    """Calculs transfert thermique en recirculation"""
    
    @staticmethod
    def calculate_temperature_evolution(
            temp_tank: float,         # °C
            temp_return: float,       # °C
            temp_ambient: float,      # °C
            power_heating: float,     # W
            time_step: float,         # s
            flow_rate: float,         # m³/h
            volume: float,            # L
            cp: float = 4186,         # J/kg.K
            rho: float = 1000,        # kg/m³
            detailed: bool = False
        ) -> Dict[str, float]:
        """
        Calcule l'évolution de température avec recirculation
        en utilisant les classes existantes
        """
        try:
            # 1. Utiliser ThermalProperties pour les propriétés de l'eau
            water_props = ThermalProperties.get_water_properties(
                temp_tank, 
                pressure=1.0,
                context="process"
            )
            cp = water_props['cp']  # Utiliser les vraies propriétés
            rho = water_props['rho']

            # 2. Calcul du flux de recirculation
            mass_flow = flow_rate * rho / 3600  # kg/s
            mass_tank = volume * rho / 1000     # kg
            q_recirc = mass_flow * cp * (temp_return - temp_tank)

            # 3. Utiliser les coefficients standards pour les pertes
            h_loss = Config.MaterialProperties.THERMAL_CONTACT['316L'].get('mineral_wool', 0.8)  # W/m².K
            area = (volume/1000) ** (2/3) * 6  # Surface approximative tank m²
            q_loss = h_loss * area * (temp_tank - temp_ambient)

            # 4. Bilan énergétique total
            q_net = power_heating + q_recirc - q_loss

            # 5. Évolution température avec limitation physique
            delta_t = (q_net * time_step) / (mass_tank * cp)
            max_delta_t = Config.CorrelationLimits.MAX_TEMP_GRADIENT * time_step 
            delta_t = np.clip(delta_t, -max_delta_t, max_delta_t)

            # 6. Nouvelle température avec validation
            temp_new = ThermalProperties.validate_process_temperature(temp_tank + delta_t)

            results = {
                'temp': temp_new,
                'delta_t': delta_t,
                'power_net': q_net,
                'power_recirc': q_recirc,
                'power_loss': q_loss
            }

            if detailed:
                results.update({
                    'mass_flow': mass_flow,
                    'heat_transfer': {
                        'h_loss': h_loss,
                        'area': area,
                        'water_properties': water_props,
                        'temp_diff': temp_tank - temp_ambient
                    }
                })

            return results

        except Exception as e:
            logger.error(f"Erreur calcul évolution température: {str(e)}")
            return {
                'temp': temp_tank,
                'delta_t': 0,
                'power_net': 0,
                'power_recirc': 0,
                'power_loss': 0
            }

    @staticmethod
    def calculate_mixing_temperature(
        volume_hot: float,      # m³
        temp_hot: float,        # °C
        volume_cold: float,     # m³
        temp_cold: float,       # °C
        Q_added: float = 0,     # W (chauffage supplémentaire)
        time_step: float = 30   # s
    ) -> Dict[str, float]:
        """
        Calcule les températures après mélange avec chauffage
        """
        props = ThermalProperties.get_water_properties((temp_hot + temp_cold)/2, 1.0)
        
        # Énergie totale système
        E_hot = volume_hot * props['rho'] * props['cp'] * temp_hot
        E_cold = volume_cold * props['rho'] * props['cp'] * temp_cold
        E_heat = Q_added * time_step
        
        # Température finale
        mass_total = (volume_hot + volume_cold) * props['rho']
        temp_final = (E_hot + E_cold + E_heat) / (mass_total * props['cp'])
        
        return {
            'temp_final': temp_final,
            'energy_total': E_hot + E_cold + E_heat
        }