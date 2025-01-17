# BIOREACTOR_DESIGN

## Description
Un projet Python complet pour la modélisation et l'analyse thermique de bioréacteurs, incluant des calculs de transfert de chaleur, le dimensionnement des systèmes CIP (Clean-In-Place), et diverses solutions de chauffage.

## État du Projet
🚧 **En développement**

- ✅ Modélisation thermique de base
- ✅ Calculs CIP basiques
- ✅ Analyses thermiques des échangeurs
- 🚧 Module de recirculation (Section IV du notebook - En cours de restructuration)

## Points d'Attention
Le module de recirculation (Partie IV du notebook) nécessite une refactorisation majeure pour :

- Améliorer la gestion des états thermiques
- Optimiser les calculs de transfert de chaleur
- Renforcer la robustesse du code
- Implémenter une meilleure validation des paramètres
- Ajouter des tests unitaires complets

## Installation
```bash
git clone https://github.com/votre-username/BIOREACTOR_DESIGN
cd BIOREACTOR_DESIGN
pip install -r requirements.txt
```
## Lancer des calculs
Un lien vers le notebook [heat_transfer_analysis.ipynb](notebook/heat_transfer_analysis.ipynb) permet d'effectuer des analyses thermiques et le dimensionnement.

## Architecture du Projet
```plaintext
BIOREACTOR_DESIGN/
├── src/
│   ├── config/
│   │   └── config.py           # Configuration globale, limites opératoires, constantes physiques
│   ├── equipment/
│   │   ├── tanks/
│   │   │   ├── bioreactor.py   # Classe principale du bioréacteur avec géométrie et propriétés thermiques
│   │   │   └── process_tank.py # Tanks process pour CIP, média, tampons
│   │   ├── heat_system/
│   │   │   ├── electrical_heater.py    # Système de chauffe électrique avec PID
│   │   │   ├── plate_exchanger.py      # Échangeur à plaques avec corrélations TEMA
│   │   │   └── steam_generator.py      # Générateur vapeur avec contrôle
│   │   ├── piping/
│   │   │   └── pipeline.py     # Calculs pertes de charge et thermiques tuyauterie
│   │   └── spray_devices/
│   │       └── rotary_jet_mixer.py    # Modèle boule de lavage rotative
│   └── process/
│       ├── thermal/
│       │   ├── cip_model.py           # Modèle CIP avec recirculation (à refactoriser)
│       │   ├── heating.py             # Calculs de chauffe multi-modes
│       │   ├── cooling.py             # Refroidissement naturel et forcé
│       │   ├── losses.py              # Pertes thermiques multi-zones
│       │   ├── thermal_calculations.py # Calculs thermiques standards
│       │   └── thermal_utils.py       # Utilitaires et propriétés physiques
│       └── mixing/
│           └── jet_impact.py          # Calculs impact jets et cisaillement
├── tests/
│   ├── test_heating_performance.py     # Tests performances chauffage
│   ├── test_heating.py                 # Tests unitaires chauffage
│   ├── test_heating_system.py          # Tests système complet
│   ├── test_plate_exchanger.py         # Tests échangeur à plaques
│   ├── test_thermal_calculations.py    # Tests calculs thermiques
│   └── test_thermal_utils.py           # Tests utilitaires thermiques
├── notebook/
│   └── heat_transfer_analysis.ipynb    # Analyses thermiques et dimensionnement
└── README.md
```

## Description détaillée des fichiers

### Configuration
- `config.py` : Centralise toutes les constantes et limites du projet incluant les paramètres process, les corrélations, les propriétés des matériaux, et les facteurs de sécurité.

### Équipements

#### Tanks
- `bioreactor.py` : Implémente la classe Bioreactor avec sa géométrie cylindro-conique, propriétés thermiques et méthodes de calcul associées.
- `process_tank.py` : Gère les tanks auxiliaires (CIP, média) avec leur dimensionnement et caractéristiques spécifiques.

#### Systèmes de chauffe
- `electrical_heater.py` : Modélisation complète des résistances avec contrôle PID, inertie thermique et sécurités.
- `plate_exchanger.py` : Échangeur à plaques selon standards TEMA avec calculs NUT/DTLM et coefficients d'échange.
- `steam_generator.py` : Générateur vapeur avec régulation et calculs énergétiques.

#### Tuyauterie & Accessoires
- `pipeline.py` : Calculs hydrauliques et thermiques des tuyauteries avec isolation.
- `rotary_jet_mixer.py` : Modèle de la boule de lavage rotative avec impacts et couverture.

### Process

#### Thermique
- `cip_model.py` : Modèle du système CIP complet (à refactoriser).
- `heating.py` : Calculs de chauffe incluant différentes méthodes et optimisations.
- `cooling.py` : Modélisation du refroidissement avec stratification.
- `losses.py` : Calcul détaillé des pertes thermiques par zones.
- `thermal_calculations.py` : Fonctions de base pour les calculs thermiques.
- `thermal_utils.py` : Utilitaires, propriétés physiques et validations.

#### Mélange
- `jet_impact.py` : Calculs des impacts jets et contraintes de cisaillement.

### Tests
Ensemble complet de tests unitaires et fonctionnels pour valider les calculs et les modèles (à refactoriser).

