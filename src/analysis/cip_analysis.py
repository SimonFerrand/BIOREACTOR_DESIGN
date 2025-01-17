"""
ANALYSE OPTIMISÉE DU SYSTÈME CIP AVEC RECIRCULATION - VERSION 2 AMÉLIORÉE
Configuration :
- Tank process: 38L avec résistance 9kW
- Fermenteur: 300L avec boule de nettoyage
- Points de contrôle: T entrée fermenteur et T retour tank
- Améliorations : PID optimisé, inertie thermique, pertes réelles
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, Any

# Imports des modules src
from src.equipment.tanks.bioreactor import TankGeometry, Bioreactor
from src.equipment.tanks.process_tank import ProcessTank
from src.equipment.heat_system.electrical_heater import create_standard_heater
from src.process.thermal.cip_model import CIPThermalModel
from src.equipment.piping.pipeline import PipeSpecs
from src.process.thermal.thermal_utils import ThermalProperties
from src.process.thermal.losses import ThermalCalculator
from src.config import Config

def setup_optimized_equipment():
    """Configuration optimisée des équipements avec isolation renforcée"""
    
    # Tank process 38L optimisé
    geometry_process = TankGeometry(
        diameter=400,         # mm
        height_cylinder=400,  # mm
        height_total=600,     # mm
        volume_useful=38,     # L
        volume_cone=2,        # L
        volume_total=40       # L
    )

    tank_process = ProcessTank(
        geometry=geometry_process,
        tank_type='heating',
        material="316L",
        insulation_type="mineral_wool",
        insulation_thickness=50.0,  # Augmenté pour réduire les pertes
        design_temperature=95.0
    )

    # Fermenteur 300L avec isolation standard
    geometry_ferm = TankGeometry(
        diameter=650,         # mm
        height_cylinder=900,  # mm 
        height_total=1500,   # mm
        volume_useful=300,   # L
        volume_cone=20,      # L
        volume_total=320     # L
    )

    fermentor = Bioreactor(
        geometry=geometry_ferm,
        material="316L",
        insulation_type="mineral_wool",
        insulation_thickness=50.0,
        design_temperature=95.0
    )

    # Résistance 9kW avec PID optimisé
    heater = create_standard_heater(9.0)

    # Tuyauterie optimisée DN25 avec isolation renforcée
    pipe_specs = {
        'supply': PipeSpecs(
            diameter=25.4,    # DN25
            length=5.0,       # m
            thickness=2.0,    # mm
            insulation_type='mineral_wool',
            insulation_thickness=50.0  # Augmenté pour réduire les pertes
        ),
        'return': PipeSpecs(
            diameter=25.4,    # DN25
            length=5.0,       # m
            thickness=2.0,    # mm
            insulation_type='mineral_wool',
            insulation_thickness=50.0
        )
    }

    return tank_process, fermentor, heater, pipe_specs

def get_test_configurations():
    """Configuration des tests selon standards BPF"""
    return {
        # Débits testés (vitesse impact 1-2 m/s)
        'flow_rates': [4.0, 4.5, 5.0],  # m³/h
        
        # Nombres de volumes selon BPF
        'n_volumes': [2.0, 2.5, 3.0],   
        
        # Consigne et paramètres opératoires
        'temp_target': 80.0,            # °C
        'pressure': 4.0,                # bar (minimum CIP)
        'time_step': 1.0,              # s (réduit pour précision)
        
        # Critères validation
        'min_contact_time': 20,         # min
        'temp_tolerance': 2.0,          # °C
        'stability_limit': 1.0          # °C
    }

def analyze_cip_results(results: Dict, config: Dict) -> Dict:
    """Analyse détaillée des résultats CIP avec calculs énergétiques"""
    
    # 1. Extraction métriques clés
    perf = results['performance']
    temps_final = results['final_temps']
    
    # 2. Analyse énergétique détaillée
    volume_total = config['flow_rate'] * (results['total_duration']/60)  # m³
    energy_per_m3 = results['total_energy'] / volume_total  # kWh/m³
    
    # Calcul rendement thermique
    if 'cip_phase' in results and 'evolution_data' in results['cip_phase']:
        power_data = np.array([d.get('heating', {}).get('power', 0) 
                             for d in results['cip_phase']['evolution_data']])
        power_losses = np.array([d.get('evolution', {}).get('power_loss', 0)
                               for d in results['cip_phase']['evolution_data']])
        thermal_efficiency = (1 - np.mean(power_losses)/np.mean(power_data)) * 100
    else:
        thermal_efficiency = None
    
    # 3. Analyse stabilité température
    temp_data = np.array(results['cip_phase']['temps_reactor_inlet'])
    temp_mean = np.mean(temp_data)
    temp_std = np.std(temp_data)
    temp_range = np.max(temp_data) - np.min(temp_data)
    
    # Analyse régime transitoire vs établi
    steady_state_idx = int(len(temp_data) * 0.2)  # Ignore premiers 20%
    temp_stability = np.std(temp_data[steady_state_idx:])
    
    # 4. Validation selon critères CIP
    temp_ok = (temps_final['reactor_inlet'] >= config['temp_target'] - 2 and
              temp_stability <= config['stability_limit'])
    time_ok = results['total_duration'] >= config['min_contact_time']
    thermal_ok = temp_range <= 5.0  # Max 5°C de variation totale
    
    return {
        'energy_metrics': {
            'total_energy': results['total_energy'],
            'energy_per_m3': energy_per_m3,
            'average_power': np.mean(results['cip_phase']['powers']),
            'thermal_efficiency': thermal_efficiency
        },
        'temperature_metrics': {
            'final_temp': temps_final['reactor_inlet'],
            'mean_temp': temp_mean,
            'stability': temp_stability,
            'range': temp_range,
            'std_dev': temp_std
        },
        'validation': {
            'temp_ok': temp_ok,
            'time_ok': time_ok,
            'thermal_ok': thermal_ok,
            'all_ok': temp_ok and time_ok and thermal_ok
        }
    }

def plot_detailed_analysis(results_matrix: list, test_configs: Dict):
    """Visualisation détaillée des résultats avec analyses statistiques"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Evolution température pour chaque configuration
    ax1 = plt.subplot(3, 2, 1)
    for result in results_matrix:
        config = result['config']
        temps = np.array(result['results']['cip_phase']['temps_reactor_inlet'])
        times = np.array(result['results']['cip_phase']['times'])
        label = f"{config['flow_rate']}m³/h, {config['n_volumes']}vol"
        ax1.plot(times, temps, label=label, alpha=0.7)
    
    ax1.axhline(y=test_configs['temp_target'], color='k', 
                linestyle='--', label='Cible')
    ax1.fill_between([0, max(times)], 
                     test_configs['temp_target'] - test_configs['temp_tolerance'],
                     test_configs['temp_target'] + test_configs['temp_tolerance'],
                     color='gray', alpha=0.2, label='Zone acceptable')
    ax1.grid(True)
    ax1.set_xlabel('Temps (min)')
    ax1.set_ylabel('Température (°C)')
    ax1.set_title('Evolution température entrée réacteur')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Analyse énergétique
    ax2 = plt.subplot(3, 2, 2)
    flow_rates = sorted(list(set([r['config']['flow_rate'] for r in results_matrix])))
    n_volumes = sorted(list(set([r['config']['n_volumes'] for r in results_matrix])))
    
    for n_vol in n_volumes:
        energies = [r['analysis']['energy_metrics']['energy_per_m3'] 
                   for r in results_matrix 
                   if r['config']['n_volumes'] == n_vol]
        ax2.plot(flow_rates, energies, 'o-', 
                label=f'{n_vol} volumes', markersize=8)
    
    ax2.grid(True)
    ax2.set_xlabel('Débit (m³/h)')
    ax2.set_ylabel('Energie (kWh/m³)')
    ax2.set_title('Consommation énergétique')
    ax2.legend()
    
    # 3. Stabilité température
    ax3 = plt.subplot(3, 2, 3)
    for n_vol in n_volumes:
        stabilities = [r['analysis']['temperature_metrics']['stability']
                      for r in results_matrix 
                      if r['config']['n_volumes'] == n_vol]
        ax3.plot(flow_rates, stabilities, 'o-',
                label=f'{n_vol} volumes', markersize=8)
    
    ax3.axhline(y=test_configs['stability_limit'], color='r',
                linestyle='--', label='Limite stabilité')
    ax3.grid(True)
    ax3.set_xlabel('Débit (m³/h)')
    ax3.set_ylabel('Stabilité (°C)')
    ax3.set_title('Stabilité température')
    ax3.legend()
    
    # 4. Analyse statistique températures
    ax4 = plt.subplot(3, 2, 4)
    box_data = []
    box_labels = []
    
    for result in results_matrix:
        temps = np.array(result['results']['cip_phase']['temps_reactor_inlet'])
        box_data.append(temps)
        box_labels.append(f"{result['config']['flow_rate']}m³/h\n{result['config']['n_volumes']}vol")
    
    ax4.boxplot(box_data, labels=box_labels)
    ax4.axhline(y=test_configs['temp_target'], color='k',
                linestyle='--', label='Cible')
    ax4.grid(True)
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('Température (°C)')
    ax4.set_title('Distribution températures')
    
    # 5. Rendement thermique
    ax5 = plt.subplot(3, 2, 5)
    for n_vol in n_volumes:
        efficiencies = [r['analysis']['energy_metrics'].get('thermal_efficiency', 0)
                       for r in results_matrix
                       if r['config']['n_volumes'] == n_vol]
        ax5.plot(flow_rates, efficiencies, 'o-',
                label=f'{n_vol} volumes', markersize=8)
    
    ax5.grid(True)
    ax5.set_xlabel('Débit (m³/h)')
    ax5.set_ylabel('Rendement (%)')
    ax5.set_title('Rendement thermique')
    ax5.legend()
    
    # 6. Temps de cycle
    ax6 = plt.subplot(3, 2, 6)
    for n_vol in n_volumes:
        durations = [r['results']['total_duration']
                    for r in results_matrix
                    if r['config']['n_volumes'] == n_vol]
        ax6.plot(flow_rates, durations, 'o-',
                label=f'{n_vol} volumes', markersize=8)
    
    ax6.axhline(y=test_configs['min_contact_time'], color='r',
                linestyle='--', label='Temps minimum')
    ax6.grid(True)
    ax6.set_xlabel('Débit (m³/h)')
    ax6.set_ylabel('Durée (min)')
    ax6.set_title('Temps de cycle')
    ax6.legend()
    
    plt.tight_layout()
    return fig

def main():
    """Exécution de l'analyse complète"""
    print("\nDÉBUT ANALYSE SYSTÈME CIP")
    print("-" * 50)
    
    # 1. Setup équipements
    tank, fermentor, heater, pipes = setup_optimized_equipment()
    
    # 2. Création système CIP
    cip_model = CIPThermalModel(
        tank_process=tank,
        reactor=fermentor,
        heater=heater,
        pipe_specs=pipes
    )
    
    # 3. Configuration tests
    test_configs = get_test_configurations()
    
    # 4. Exécution des tests
    results_matrix = []
    
    print("\nExécution des simulations:")
    for flow_rate in test_configs['flow_rates']:
        for n_vol in test_configs['n_volumes']:
            print(f"\nTest: {flow_rate} m³/h, {n_vol} volumes")
            
            config = {
                'flow_rate': flow_rate,
                'n_volumes': n_vol,
                'temp_target': test_configs['temp_target'],
                'pressure': test_configs['pressure'],
                'detailed': True
            }
            
            try:
                results = cip_model.calculate_cip_cycle(**config)
                analysis = analyze_cip_results(results, test_configs)
                
                results_matrix.append({
                    'config': config,
                    'results': results,
                    'analysis': analysis
                })
                
                print(f"OK - Durée: {results['total_duration']:.1f}min")
                
            except Exception as e:
                print(f"Erreur: {str(e)}")
    
    
    # 5. Visualisation résultats
    if results_matrix:
        fig = plot_detailed_analysis(results_matrix, test_configs)
        
        # 6. Analyse des meilleures configurations
        print("\nANALYSE DES CONFIGURATIONS")
        print("-" * 50)
        
        # Pour chaque critère
        best_configs = {
            'energy': min(results_matrix, 
                         key=lambda x: x['analysis']['energy_metrics']['energy_per_m3']),
            'stability': min(results_matrix, 
                           key=lambda x: x['analysis']['temperature_metrics']['stability']),
            'time': min(results_matrix, 
                       key=lambda x: x['results']['total_duration'])
        }

        for metric, result in best_configs.items():
            config = result['config']
            analysis = result['analysis']
            
            print(f"\nMeilleure configuration pour {metric}:")
            print(f"- Débit: {config['flow_rate']} m³/h")
            print(f"- Volumes: {config['n_volumes']}")
            print(f"- Énergie: {analysis['energy_metrics']['energy_per_m3']:.3f} kWh/m³")
            print(f"- Stabilité: ±{analysis['temperature_metrics']['stability']:.1f}°C")
            print(f"- Durée: {result['results']['total_duration']:.1f} min")
            if 'thermal_efficiency' in analysis['energy_metrics']:
                print(f"- Rendement: {analysis['energy_metrics']['thermal_efficiency']:.1f}%")
        
        # 7. Configuration optimale validée
        valid_configs = [r for r in results_matrix 
                        if r['analysis']['validation']['all_ok']]
        
        if valid_configs:
            # Sélection configuration optimale en pondérant les critères
            def calculate_score(result):
                analysis = result['analysis']
                config = result['config']
                
                # Normalisation des métriques
                energy_score = analysis['energy_metrics']['energy_per_m3'] / 2.0  # Base 2 kWh/m³
                stability_score = analysis['temperature_metrics']['stability']
                time_score = result['results']['total_duration'] / 30  # Base 30 min
                
                # Pondération (ajustable selon priorités)
                return (0.4 * energy_score + 
                       0.4 * stability_score + 
                       0.2 * time_score)
            
            best_valid = min(valid_configs, key=calculate_score)
            
            print("\nCONFIGURATION OPTIMALE RECOMMANDÉE:")
            print("-" * 50)
            print(f"Débit: {best_valid['config']['flow_rate']} m³/h")
            print(f"Volumes: {best_valid['config']['n_volumes']}")
            print("\nPerformances attendues:")
            print(f"- Énergie: {best_valid['analysis']['energy_metrics']['energy_per_m3']:.3f} kWh/m³")
            print(f"- Stabilité: ±{best_valid['analysis']['temperature_metrics']['stability']:.1f}°C")
            print(f"- Durée: {best_valid['results']['total_duration']:.1f} min")
            print(f"- Rendement: {best_valid['analysis']['energy_metrics'].get('thermal_efficiency', 0):.1f}%")
            
            # 8. Analyse économique
            energy_cost = 0.15  # €/kWh
            volume_annual = 300 * 250  # L/an (300L/jour, 250 jours)
            annual_cost = (best_valid['analysis']['energy_metrics']['energy_per_m3'] * 
                         volume_annual/1000 * energy_cost)
            
            print("\nIMPACT ÉCONOMIQUE (estimation):")
            print(f"- Coût énergétique annuel: {annual_cost:.2f} €")
            print(f"- Coût par cycle: {annual_cost/250:.2f} €")
            
        else:
            print("\nAUCUNE CONFIGURATION NE SATISFAIT TOUS LES CRITÈRES!")
            print("Révision nécessaire des paramètres ou de l'équipement.")
        
        return fig
    else:
        print("\nAUCUN RÉSULTAT DISPONIBLE!")
        return None

if __name__ == "__main__":
    # Configuration matplotlib
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = [15, 10]
    
    # Exécution analyse
    fig = main()
    
    # Affichage résultats
    if fig:
        plt.show()