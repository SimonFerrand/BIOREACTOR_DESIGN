import unittest
from src.process.thermal.thermal_calculations import ThermalCalculations
from src.config import Config

class TestThermalCalculations(unittest.TestCase):
    def test_lmtd_calculation(self):
        """Test du calcul DTLM pour différents cas"""
        # Cas nominal contre-courant
        dtlm = ThermalCalculations.calculate_lmtd(
            t_hot_in=127.4,   # Vapeur entrée
            t_hot_out=127.4,  # Vapeur sortie (condensation)
            t_cold_in=10.0,   # Eau entrée
            t_cold_out=70.0   # Eau sortie
        )
        self.assertTrue(
            Config.PlateExchangerLimits.LMTD_MIN <= dtlm <= 100,
            f"DTLM {dtlm:.1f}°C hors limites typiques"
        )

    def test_condensation_coefficient(self):
        """Test du calcul du coefficient de condensation"""
        h_cond = ThermalCalculations.calculate_condensation_coeff(
            temp=127.4,    # °C
            pressure=2.5,  # bar
            height=0.5     # m
        )
        self.assertTrue(
            Config.PlateExchangerLimits.HTC_MIN <= h_cond <= \
            Config.PlateExchangerLimits.STEAM_CONDENSING_MAX,
            f"Coefficient condensation {h_cond:.0f} W/m².K hors limites"
        )

    def test_water_coefficient(self):
        """Test du coefficient convection eau"""
        h_water = ThermalCalculations.calculate_water_coeff(
            flow_rate=6.0,  # m³/h
            temp=40.0,     # °C
            dh=0.005       # m
        )
        self.assertTrue(
            Config.PlateExchangerLimits.WATER_MIN <= h_water <= \
            Config.PlateExchangerLimits.WATER_MAX,
            f"Coefficient eau {h_water:.0f} W/m².K hors limites"
        )

    def test_extreme_cases(self):
        """Test des cas limites"""
        # DTLM trop faible
        with self.assertRaises(ValueError):
            ThermalCalculations.calculate_lmtd(
                t_hot_in=50.0,
                t_hot_out=45.0,
                t_cold_in=40.0,
                t_cold_out=42.0
            )

        # Débit trop faible
        with self.assertRaises(ValueError):
            ThermalCalculations.calculate_water_coeff(
                flow_rate=0.1,  # trop faible
                temp=20.0,
                dh=0.005
            )

if __name__ == '__main__':
    unittest.main()