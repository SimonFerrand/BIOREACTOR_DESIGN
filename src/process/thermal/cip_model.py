# src/process/thermal/cip_model.py
from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import logging
from src.config import Config
from src.process.thermal.thermal_utils import ThermalProperties
from src.process.thermal.heating import HeatingCalculator
from src.equipment.piping.pipeline import Pipeline, PipeSpecs
from src.process.thermal.losses import ThermalCalculator
from src.process.thermal.thermal_calculations import CirculationHeatTransfer

# Configuration du logging
logger = logging.getLogger(__name__)

@dataclass
class ProcessTemperatures:
    """Structure pour stocker les températures process"""
    tank: float             # °C (température tank)
    tank_return: float      # °C (température retour tank - protection résistance)
    reactor_inlet: float    # °C (température entrée réacteur - point de contrôle)

class CIPThermalModel:
    """
    Modèle thermique pour système CIP avec boule de nettoyage
    Prend en compte:
    - Chauffage tank process
    - Température d'impact sur parois
    - Circuit de recirculation avec pertes
    """
    def __init__(
        self,
        tank_process,
        reactor,
        heater,
        pipe_specs: Optional[Dict[str, PipeSpecs]] = None
    ):
        self.tank_process = tank_process
        self.reactor = reactor
        self.heater = heater
        
        # Création circuit par défaut si non spécifié
        if pipe_specs is None:
            pipe_specs = {
                'supply': PipeSpecs(
                    diameter=25.4,    # DN25
                    length=5.0,       # m
                    thickness=2.0,    # mm
                    insulation_type='mineral_wool',
                    insulation_thickness=25.0
                ),
                'return': PipeSpecs(
                    diameter=25.4,
                    length=5.0,
                    thickness=2.0,
                    insulation_type='mineral_wool',
                    insulation_thickness=25.0
                )
            }
        
        # Création tuyauterie
        self.pipes = {
            'supply': Pipeline(
                specs=pipe_specs['supply'],
                material="316L",
                design_pressure=6.0,
                design_temperature=95.0
            ),
            'return': Pipeline(
                specs=pipe_specs['return'],
                material="316L",
                design_pressure=6.0,
                design_temperature=95.0
            )
        }
        
        # Volume circuit
        self.circuit_volume = sum(p.volume for p in self.pipes.values())
        
        # Calculateurs thermiques
        self.heating_calc = HeatingCalculator(tank_process)
        self.thermal_calc = ThermalCalculator(tank_process)
             
    def simulate_cip_phase(
            self,
            temp_initial: Dict[str, float],
            temp_target: float,
            flow_rate: float,      # m³/h
            duration: float,       # min
            time_step: float = 30.0, # Retour à 30s 
            pressure: float = 4.0,
            detailed: bool = False
        ) -> Dict:
            """Simule la phase CIP avec modèle simplifié"""
            try:
                # 1. Validation et initialisation
                Config.validate_operating_conditions(
                    max(temp_initial.values()),
                    pressure,
                    mode='cip'
                )
                
                state = ProcessTemperatures(
                    tank=ThermalProperties.validate_process_temperature(temp_initial['tank']),
                    tank_return=ThermalProperties.validate_process_temperature(temp_initial['tank']),  
                    reactor_inlet=ThermalProperties.validate_process_temperature(temp_initial['tank'])
                )

                # 2. Paramètres du circuit
                volume_total = self.tank_process.geometry.volume_useful + self.circuit_volume
                transit_time = self.circuit_volume / (flow_rate * 1000 / 3600)  # s

                # 3. Initialisation résultats
                results = {
                    'times': [],
                    'temps_tank': [],
                    'temps_tank_return': [],
                    'temps_reactor_inlet': [],
                    'powers': [],
                    'energy_consumption': 0.0
                }

                # 4. Boucle de simulation
                for t in np.arange(0, duration*60, time_step):
                    # 4.1 Calcul pertes globales
                    props = ThermalProperties.get_water_properties(
                        temp=state.tank,
                        pressure=pressure,
                        context="process"
                    )
                    
                    mass_flow = flow_rate * props['rho'] / 3600  # kg/s
                    
                    total_losses = self.thermal_calc.calculate_heat_loss(
                        temp_fluid=state.tank,
                        temp_ambient=20.0,
                        flow_rate=flow_rate,
                        pressure=pressure
                    )

                    # 4.2 Contrôle chauffage
                    heating = self.heater.calculate_heating_power_with_control(
                        temp_fluid=state.tank_return,
                        temp_target=temp_target,
                        temp_measured=state.reactor_inlet,
                        time_step=time_step
                    )

                    # 4.3 Bilan thermique global
                    q_net = heating['power']*1000 - total_losses['total_loss']  # W
                    delta_t = q_net * time_step / (volume_total * props['rho'] * props['cp'])

                    # 4.4 Mise à jour températures avec délai simplifié
                    state.tank += delta_t
                    if t >= transit_time:
                        state.reactor_inlet = state.tank - 1.0  # Perte fixe 1°C dans circuit
                        state.tank_return = state.reactor_inlet - 0.5  # Perte 0.5°C dans réacteur

                    # 4.5 Enregistrement résultats
                    results['times'].append(t/60)  # min
                    results['temps_tank'].append(state.tank)
                    results['temps_tank_return'].append(state.tank_return)
                    results['temps_reactor_inlet'].append(state.reactor_inlet)
                    results['powers'].append(heating['power'])

                    # 4.6 Calcul énergie
                    if len(results['powers']) > 1:
                        dt_h = time_step / 3600  # s -> h
                        power_avg = (results['powers'][-1] + results['powers'][-2]) / 2  
                        results['energy_consumption'] += power_avg * dt_h  # kWh

                return results
                
            except Exception as e:
                logger.error(f"Erreur simulation CIP: {str(e)}")
                raise

    def calculate_cip_cycle(
        self,
        temp_initial: float = 20.0,    # °C
        temp_target: float = 80.0,     # °C
        flow_rate: float = 4.0,        # m³/h 
        n_volumes: float = 3.0,        # Nombre de volumes à recirculer
        pressure: float = 4.0,         # bar
        detailed: bool = False
    ) -> Dict:
        """
        Calcule cycle CIP complet:
        1. Chauffe initiale tank process
        2. Recirculation avec contrôle température impact
        """
        # 1. Phase chauffe initiale
        print("\nPhase 1: Chauffe initiale du tank process")
        heating_results = self.heating_calc.calculate_electrical_heating_profile(
            heater=self.heater,
            temp_initial=temp_initial,
            temp_target=temp_target,
            detailed=detailed
        )
        
        # 2. Phase recirculation
        volume_reactor = self.reactor.geometry.volume_useful  # L
        duration = (n_volumes * volume_reactor / 1000) / flow_rate * 60  # min
        
        print(f"\nPhase 2: Recirculation ({n_volumes:.1f} volumes)")
        print(f"- Volume réacteur: {volume_reactor}L")
        print(f"- Débit: {flow_rate}m³/h")
        print(f"- Durée: {duration:.1f}min")
        
        cip_results = self.simulate_cip_phase(
            temp_initial={
                'tank': heating_results['temperatures'][-1],
                'tank_return': heating_results['temperatures'][-1],
                'reactor_inlet': heating_results['temperatures'][-1]
            },
            temp_target=temp_target,
            flow_rate=flow_rate,
            duration=duration,
            pressure=pressure,
            detailed=detailed
        )

        
        # 3. Compilation résultats
        results = {
            'heating_phase': heating_results,
            'cip_phase': cip_results,
            'total_energy': (
                heating_results.get('total_energy', 0) +
                cip_results['energy_consumption']
            ),
            'total_duration': (
                heating_results['duration'] +
                duration
            ),
            'final_temps': {
                'tank': cip_results['temps_tank'][-1],
                'reactor_inlet': cip_results['temps_reactor_inlet'][-1],
                'tank_return': cip_results['temps_tank_return'][-1]
            }
        }
        
        # Correction du calcul de l'énergie totale
        total_energy = heating_results.get('total_energy', 0) + cip_results['energy_consumption']
        energy_per_volume = total_energy / (volume_reactor / 1000)  # kWh/m³ -> kWh/L

        if detailed:
            # Performance metrics
            props_avg = ThermalProperties.get_water_properties(results['final_temps']['tank'], pressure)
            
            results['performance'] = {
                'energy_per_volume': total_energy / (volume_reactor / 1000),  # Correction unités kWh/L
                'impact_temp_stability': np.std(cip_results['temps_reactor_inlet']),
                'process_efficiency': min(
                    results['final_temps']['reactor_inlet'] / temp_target * 100,
                    100.0
                ),
                'power_mean': np.mean(cip_results['powers']) if cip_results['powers'] else 0,  # Ajout
                'fluid_properties': props_avg,
                'process_data': {
                    'flow_rate': flow_rate,
                    'n_volumes': n_volumes,
                    'volume_reactor': volume_reactor,
                    'circuit_volume': self.circuit_volume,
                    'pressure': pressure
                }
            }
                
        return results