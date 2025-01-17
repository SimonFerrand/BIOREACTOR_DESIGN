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
            time_step: float = 1.0,  # s (réduit pour meilleur contrôle)
            pressure: float = 4.0, # bar
            detailed: bool = False
        ) -> Dict:
            """
            Simule la phase de recirculation CIP avec contrôle sur température d'impact
            """
            try:
                # 1. Validation paramètres
                Config.validate_operating_conditions(
                    max(temp_initial.values()),
                    pressure,
                    mode='cip'
                )
                
                # 2. Initialisation état avec validation
                state = ProcessTemperatures(
                    tank=ThermalProperties.validate_process_temperature(temp_initial['tank']),
                    tank_return=ThermalProperties.validate_process_temperature(temp_initial['tank']),  
                    reactor_inlet=ThermalProperties.validate_process_temperature(temp_initial['tank'])
                )

                # 3. Calcul des temps de transit avec prise en compte du régime d'écoulement
                volume_supply = self.pipes['supply'].volume  # L
                volume_return = self.pipes['return'].volume  # L
                
                # Calcul Reynolds pour correction temps transit
                area_pipe = np.pi * (self.pipes['supply'].specs.diameter/1000/2)**2
                velocity = (flow_rate/3600) / area_pipe
                re = ThermalProperties.calculate_reynolds(
                    velocity=velocity,
                    length=self.pipes['supply'].specs.diameter/1000,
                    temp=temp_initial['tank'],
                    pressure=pressure,
                    context="pipe"
                )
                
                # Facteur correction temps transit selon régime
                transit_factor = 0.8 if re > Config.CorrelationLimits.RE_MIN_TURB else 1.2
                transit_time_supply = (volume_supply / (flow_rate * 1000 / 3600)) * transit_factor
                transit_time_return = (volume_return / (flow_rate * 1000 / 3600)) * transit_factor
                
                # 4. Ajuster time_step pour avoir au moins 10 points pendant le transit
                time_step = min(time_step, transit_time_supply / 10)
                
                # 5. Configuration logging
                print("\n=== Début Simulation CIP ===")
                print(f"Conditions: T_target={temp_target}°C, Flow={flow_rate}m³/h, Re={re:.0f}")
                print(f"Volumes: supply={volume_supply:.1f}L, return={volume_return:.1f}L")
                print(f"Transit: supply={transit_time_supply:.1f}s, return={transit_time_return:.1f}s")
                print(f"Time step: {time_step:.1f}s")

                # 6. Initialisation résultats
                results = {
                    'times': [],
                    'temps_tank': [],
                    'temps_tank_return': [],
                    'temps_reactor_inlet': [],
                    'powers': [],
                    'energy_consumption': 0.0,
                    'control_info': []
                }

                # Files pour les délais de transport
                history_supply = []
                history_return = []

                # 7. Boucle de simulation
                for t in np.arange(0, duration*60, time_step):
                    print(f"\nPas {t/60:.1f}min:")
                    print(f"État: Tank={state.tank:.1f}°C, Return={state.tank_return:.1f}°C, Inlet={state.reactor_inlet:.1f}°C")

                    # 7.1 Calcul précis des propriétés
                    props = ThermalProperties.get_water_properties(
                        temp=state.tank,
                        pressure=pressure,
                        context="process"
                    )
                    mass_flow = flow_rate * props['rho'] / 3600  # kg/s

                    # 7.2 Pertes ligne supply avec modèle complet
                    losses_supply = self.pipes['supply'].calculate_heat_loss(
                        state.tank, 20.0, flow_rate, pressure
                    )
                    delta_t_supply = losses_supply['heat_loss'] / (mass_flow * props['cp'])
                    temp_after_supply = state.tank - delta_t_supply
                    
                    print(f"Supply: Pertes={losses_supply['heat_loss']/1000:.2f}kW, T_après={temp_after_supply:.1f}°C")
                    
                    # 7.3 Mise à jour reactor_inlet avec délai
                    steps_delay_supply = max(1, int(transit_time_supply / time_step))
                    history_supply.append(temp_after_supply)
                    
                    if len(history_supply) >= steps_delay_supply:
                        state.reactor_inlet = history_supply.pop(0)
                        while len(history_supply) > steps_delay_supply:
                            history_supply.pop(0)
                    
                    print(f"Délai supply: steps={steps_delay_supply}, T_inlet={state.reactor_inlet:.1f}°C")
                    
                    # 7.4 Pertes réacteur avec modèle complet
                    reactor_losses = self.thermal_calc.calculate_heat_loss(
                        temp_fluid=state.reactor_inlet,
                        temp_ambient=20.0,
                        flow_rate=flow_rate,
                        pressure=pressure,
                        detailed=True
                    )
                    delta_t_reactor = reactor_losses['total_loss'] / (mass_flow * props['cp'])
                    temp_after_reactor = state.reactor_inlet - delta_t_reactor
                    
                    print(f"Réacteur: Pertes={reactor_losses['total_loss']/1000:.2f}kW, T_après={temp_after_reactor:.1f}°C")
                    
                    # 7.5 Mise à jour tank_return avec délai
                    steps_delay_return = max(1, int(transit_time_return / time_step))
                    history_return.append(temp_after_reactor)
                    
                    if len(history_return) >= steps_delay_return:
                        state.tank_return = history_return.pop(0)
                        while len(history_return) > steps_delay_return:
                            history_return.pop(0)
                    
                    # 7.6 Contrôle chauffage avec PID optimisé
                    heating_power = self.heater.calculate_heating_power_with_control(
                        temp_fluid=state.tank_return,
                        temp_target=temp_target,
                        temp_measured=state.reactor_inlet,
                        margin=2.0,
                        time_step=time_step,
                        detailed=True
                    )
                    
                    # 7.7 Evolution température tank avec recirculation
                    tank_evolution = CirculationHeatTransfer.calculate_temperature_evolution(
                        temp_tank=state.tank,
                        temp_return=state.tank_return,
                        temp_ambient=20.0,
                        power_heating=heating_power['power'] * 1000,  # kW -> W
                        time_step=time_step,
                        flow_rate=flow_rate,
                        volume=self.tank_process.geometry.volume_useful,
                        cp=props['cp'],
                        rho=props['rho'],
                        detailed=True
                    )
                    
                    state.tank = tank_evolution['temp']
                    print(f"Contrôle: error={heating_power['control_info']['error']:.2f}°C")
                    print(f"Tank: deltaT={tank_evolution['delta_t']:.2f}°C, P_net={tank_evolution['power_net']/1000:.2f}kW")

                    # 7.8 Enregistrement résultats
                    results['times'].append(t/60)
                    results['temps_tank'].append(state.tank)
                    results['temps_tank_return'].append(state.tank_return)
                    results['temps_reactor_inlet'].append(state.reactor_inlet)
                    results['powers'].append(heating_power['power'])

                    # Calcul énergie avec moyenne glissante
                    if len(results['powers']) > 1:
                        dt_h = time_step / 3600  # s -> h
                        power_avg = (results['powers'][-1] + results['powers'][-2]) / 2  # kW
                        results['energy_consumption'] += power_avg * dt_h  # kWh

                    if detailed:
                        results['control_info'].append(heating_power.get('control_info', {}))

                # 8. Compilation résultats détaillés
                if detailed:
                    results.update({
                        'losses_supply': losses_supply,
                        'losses_reactor': reactor_losses,
                        'temperatures_final': state,
                        'flow_characteristics': {
                            'reynolds': re,
                            'velocity': velocity
                        },
                        'simulation_params': {
                            'time_step': time_step,
                            'transit_times': {
                                'supply': transit_time_supply,
                                'return': transit_time_return,
                                'factor': transit_factor
                            }
                        }
                    })
                    
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