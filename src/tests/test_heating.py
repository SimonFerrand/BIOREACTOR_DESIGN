import unittest
from src.process.thermal.heating import HeatingCalculator
from src.equipment.heat_system.plate_exchanger import create_standard_exchanger
from src.equipment.heat_system.steam_generator import create_standard_generator
from src.equipment.tanks.bioreactor import Bioreactor, TankGeometry

class TestHeating(unittest.TestCase):
    def setUp(self):
        """Initialisation avec équipements standards"""
        # Tank CIP 500L
        self.geometry = TankGeometry(
            diameter=800,         # mm
            height_cylinder=900,  # mm
            height_total=1200,    # mm
            volume_useful=500,    # L
            volume_cone=50,       # L
            volume_total=550      # L
        )
        
        self.tank = Bioreactor(
            geometry=self.geometry,
            material="316L",
            insulation_type="mineral_wool",
            insulation_thickness=100.0,  # mm
            design_temperature=95.0     # °C
        )
        
        # Équipements chauffage
        self.exchanger = create_standard_exchanger('6HL')    # 56 kW
        self.generator = create_standard_generator('TD13')    # 13.2 kW
        
        self.heating_calc = HeatingCalculator(self.tank)

    def test_heating_profile(self):
        """Test du profil de chauffe nominal"""
        results = self.heating_calc.calculate_heating_profile(
            exchanger=self.exchanger,
            generator=self.generator,
            temp_initial=15.0,     # °C
            temp_target=80.0,      # °C
            flow_rate=2.0,         # m³/h
            detailed=True
        )
        
        # Vérifications
        self.assertLess(results['duration'], 240,  # max 4h
                       "Temps de chauffe trop long")
        
        self.assertGreater(results['final_temp'], 70.0,
                          "Température finale trop basse")
        
        self.assertLess(np.mean(results['losses']), 1.0,
                       "Pertes thermiques trop élevées")

    def test_exchanger_performance(self):
        """Test performances échangeur"""
        perf = self.heating_calc.calculate_exchanger_performance(
            exchanger=self.exchanger,
            flow_rate=2.0,          # m³/h
            temp_in=15.0,           # °C
            temp_steam=127.4,       # °C
            pressure=2.5,           # bar
            detailed=True
        )
        
        self.assertGreater(perf['efficiency'], 0.6,
                          "Efficacité échangeur trop faible")
        
        self.assertLess(perf['power']/1000, self.generator.specs.power,
                       "Puissance dépasse générateur")

    def test_temperature_evolution(self):
        """Test évolution température avec pertes"""
        evolution = self.heating_calc.calculate_temperature_evolution(
            temp_initial=70.0,    # °C
            temp_ambient=20.0,    # °C
            power_heating=10000,  # W
            time_step=30,         # s
            flow_rate=2.0         # m³/h
        )
        
        self.assertGreater(evolution['temp'], 70.0,
                          "Pas d'augmentation température")
        
        self.assertLess(evolution['losses']['total_loss'], 1000,
                       "Pertes thermiques trop élevées")

if __name__ == '__main__':
    unittest.main()