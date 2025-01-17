import unittest
import numpy as np
from src.equipment.tanks.bioreactor import TankGeometry, Bioreactor
from src.equipment.heat_system.plate_exchanger import create_standard_exchanger
from src.equipment.heat_system.steam_generator import create_standard_generator
from src.process.thermal.heating import HeatingCalculator

class TestHeatingPerformance(unittest.TestCase):
    def setUp(self):
        """Initialisation des équipements test"""
        # Tank 500L
        geometry = TankGeometry(
            diameter=800,
            height_cylinder=900,
            height_total=1200,
            volume_useful=500,
            volume_cone=50,
            volume_total=550
        )
        
        self.tank = Bioreactor(
            geometry=geometry,
            material="316L",
            insulation_type="mineral_wool",
            insulation_thickness=100.0,
            design_temperature=95.0
        )
        
        self.exchanger = create_standard_exchanger('6HL')
        self.generator = create_standard_generator('TD13')
        self.heating_calc = HeatingCalculator(self.tank)

    def test_exchanger_performance(self):
        """Test performances échangeur à différentes températures"""
        temps = [20, 40, 60, 70, 75]
        
        for t_in in temps:
            with self.subTest(temp=t_in):
                perf = self.heating_calc.calculate_exchanger_performance(
                    exchanger=self.exchanger,
                    flow_rate=2.0,
                    temp_in=t_in,
                    temp_steam=127.4,
                    pressure=2.5,
                    detailed=True
                )
                
                print(f"\nTempérature entrée: {t_in}°C")
                print(f"- U: {perf['U']:.0f} W/m².K")
                print(f"- h vapeur: {perf['h_steam']:.0f} W/m².K")
                print(f"- h eau: {perf['h_water']:.0f} W/m².K")
                print(f"- Puissance: {perf['power']/1000:.1f} kW")
                print(f"- Q max: {perf['q_max']/1000:.1f} kW")
                
                # Vérifications
                self.assertGreater(perf['power'], 0,
                                 "Puissance échangeur nulle")
                self.assertGreater(perf['U'], Config.PlateExchangerLimits.HTC_MIN,
                                 "Coefficient U trop faible")

if __name__ == '__main__':
    unittest.main()