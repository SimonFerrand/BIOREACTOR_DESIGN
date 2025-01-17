# Test du système de chauffage
def test_heating_system_corrected():
    # 1. Création du tank de 500L avec isolation
    geometry = TankGeometry(
        diameter=800,         # mm
        height_cylinder=900,  # mm
        height_total=1200,    # mm
        volume_useful=500,    # L
        volume_cone=50,       # L
        volume_total=550      # L
    )
    
    tank = Bioreactor(
        geometry=geometry,
        material="316L",
        insulation_type="mineral_wool",
        insulation_thickness=100.0,  # mm
        design_temperature=95.0
    )

    # 2. Création des équipements
    exchanger = create_standard_exchanger('6HL')
    generator = create_standard_generator('TD13')

    # 3. Création calculateur
    heating_calc = HeatingCalculator(tank)

    # 4. Test chauffage avec paramètres corrects
    try:
        results = heating_calc.calculate_heating_profile(
            exchanger=exchanger,
            generator=generator,
            temp_initial=9.0,      # température initiale plus réaliste
            temp_target=80.0,      # température cible CIP
            flow_rate=2.0,         # débit correct
            pressure=2.0,          # pression minimum CIP
            detailed=True
        )

        print("\nTest avec paramètres corrects:")
        print(f"Température finale: {results['final_temp']:.1f}°C")
        print(f"Durée: {results['duration']:.1f} minutes")
        print(f"Puissance moyenne: {results['average_power']:.1f} kW")
        print(f"Pertes moyennes: {np.mean(results['losses']):.1f} kW")
        
        return results
        
    except Exception as e:
        print(f"Erreur lors du test: {str(e)}")
        raise

# Exécution du test corrigé
results = test_heating_system_corrected()

# Visualisation
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(results['times'], results['temperatures'], 'b-', label='Température')
plt.axhline(y=80, color='r', linestyle='--', label='Cible')
plt.grid(True)
plt.xlabel('Temps (min)')
plt.ylabel('Température (°C)')
plt.title('Evolution température')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(results['times'], results['powers'], label='Puissance')
plt.plot(results['times'], results['losses'], label='Pertes')
plt.grid(True)
plt.xlabel('Temps (min)')
plt.ylabel('Puissance (kW)')
plt.title('Evolution puissance et pertes')
plt.legend()

plt.tight_layout()