import unittest
import numpy as np
from src.process.thermal.thermal_utils import ThermalProperties
from src.config import Config

class TestThermalUtils(unittest.TestCase):
    def test_water_properties(self):
        """Test calcul propriétés eau à différentes températures"""
        # Cas nominal (20°C)
        props = ThermalProperties.get_water_properties(
            temp=20.0,
            pressure=1.0,
            context="process"
        )
        
        self.assertAlmostEqual(props['rho'], 998.0, delta=5,
                              "Masse volumique eau incorrecte")
        self.assertAlmostEqual(props['cp'], 4186.0, delta=10,
                              "Capacité thermique incorrecte")
        
        # Test haute température process
        props_hot = ThermalProperties.get_water_properties(
            temp=80.0,
            pressure=2.5,
            context="process"
        )
        
        self.assertLess(props_hot['rho'], props['rho'],
                       "Masse volumique devrait diminuer avec T")
        
        # Test contexte échangeur
        props_exchanger = ThermalProperties.get_water_properties(
            temp=100.0,  # Au-dessus limite process
            pressure=2.5,
            context="exchanger"
        )
        
        self.assertTrue(all(v > 0 for v in props_exchanger.values()),
                       "Propriétés physiques négatives")

    def test_steam_properties(self):
        """Test propriétés vapeur saturée"""
        props = ThermalProperties.get_steam_properties(pressure=2.5)
        
        self.assertAlmostEqual(props['temperature'], 127.4, delta=0.5,
                              "Température saturation incorrecte")
        self.assertTrue(2000 < props['h_vaporization'] < 2500,
                       "Enthalpie vaporisation hors plage")

    def test_dimensionless_numbers(self):
        """Test calcul nombres adimensionnels"""
        # Reynolds
        re = ThermalProperties.calculate_reynolds(
            velocity=1.0,    # m/s
            length=0.005,    # m
            temp=20.0,      # °C
            pressure=1.0,    # bar
            context="convection"
        )
        
        self.assertTrue(Config.CorrelationLimits.RE_MIN < re < \
                       Config.CorrelationLimits.RE_MAX_TURB,
                       "Reynolds hors limites physiques")
        
        # Nusselt natural
        nu = ThermalProperties.calculate_nusselt_natural(
            grashof=1e6,
            prandtl=7.0,
            geometry='vertical'
        )
        
        self.assertGreater(nu, Config.CorrelationLimits.NU_MIN,
                          "Nusselt trop faible")

    def test_heat_exchanger_ntu(self):
        """Test méthode NUT échangeur"""
        efficiency, UA = ThermalProperties.calculate_heat_exchanger_ntu(
            flow_rate=2.0,      # m³/h
            surface=1.54,       # m² (6HL)
            temp_hot=127.4,     # °C
            temp_cold=20.0,     # °C
            pressure=2.5,       # bar
            k=3000,            # W/m².K
            fouling=0.9
        )
        
        self.assertTrue(0 <= efficiency <= 1,
                       "Efficacité hors limites physiques")
        self.assertGreater(UA, 100,
                          "Coefficient d'échange global trop faible")

    def test_error_cases(self):
        """Test gestion des erreurs"""
        # Température hors limites process
        with self.assertRaises(ValueError):
            ThermalProperties.get_water_properties(
                temp=150.0,
                pressure=1.0,
                context="process"
            )
        
        # Pression négative
        with self.assertRaises(ValueError):
            ThermalProperties.get_steam_properties(pressure=-1.0)

if __name__ == '__main__':
    unittest.main()