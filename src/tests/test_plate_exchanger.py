import unittest
from src.equipment.heat_system.plate_exchanger import PlateHeatExchanger, create_standard_exchanger
from src.config import Config

class TestPlateHeatExchanger(unittest.TestCase):
    def setUp(self):
        """Initialise les tests avec un échangeur 6HL standard"""
        self.exchanger = create_standard_exchanger('6HL')
        
    def test_nominal_case(self):
        """Test cas nominal d'après documentation"""
        power, temp_out = self.exchanger.calculate_heat_transfer(
            flow_rate_primary=2.0,    # m³/h
            temp_in_primary=127.4,    # °C (2.5 bar)
            temp_out_primary=127.4,   # °C
            flow_rate_secondary=6.0,  # m³/h
            temp_in_secondary=10.0    # °C
        )
        
        # Vérification puissance nominale (±10%)
        self.assertAlmostEqual(power, 56.0, delta=5.6)
        
        # Vérification température sortie cohérente
        self.assertTrue(10 < temp_out < 95)
        
    def test_velocity_limits(self):
        """Test limites de vitesse"""
        with self.assertLogs(level='WARNING'):
            self.exchanger.calculate_heat_transfer(
                flow_rate_primary=0.1,    # Trop faible
                temp_in_primary=127.4,
                temp_out_primary=127.4,
                flow_rate_secondary=6.0,
                temp_in_secondary=10.0
            )
            
    def test_dtlm_minimum(self):
        """Test DTLM minimum"""
        with self.assertLogs(level='WARNING'):
            self.exchanger.calculate_heat_transfer(
                flow_rate_primary=2.0,
                temp_in_primary=30.0,     # Delta T trop faible
                temp_out_primary=30.0,
                flow_rate_secondary=6.0,
                temp_in_secondary=25.0
            )

if __name__ == '__main__':
    unittest.main()