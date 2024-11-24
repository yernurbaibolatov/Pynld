"""
Unit tests for core functionalities.
"""

import unittest
import numpy as np
from pynld.core import DynamicalSystem, IntegrationParameters

class TestDynamicalSystem(unittest.TestCase):

    def setUp(self):
        """
        Set up a simple 2D linear dynamical system for testing.
        Example system:
            dx/dt = y
            dy/dt = -p1 * y - p2 * x
        """
        self.parameters = {'p1': 0.01, 'p2': 1.0}
        self.initial_conditions = {'x': 1.0, 'y': 0.0}
        self.t0 = 0.0

        # Define the system function
        def system(t, state_vector, p):
            x, y = state_vector
            p1, p2 = p
            xdot = y
            ydot = -p1 * y - p2 * x
            return np.array([xdot, ydot])

        self.system = system
        self.integration_params = IntegrationParameters(
            solver='RK45', time_step=1e-5, accuracy=1e-5)
        
        # Create a DynamicalSystem instance
        self.dyn_sys = DynamicalSystem(
            system=self.system,
            t0=self.t0,
            x0=self.initial_conditions,
            parameters=self.parameters,
            integration_params=self.integration_params
        )

    def test_initialization(self):
        """Test initialization of the DynamicalSystem."""
        self.assertEqual(self.dyn_sys.t, self.t0)
        np.testing.assert_array_almost_equal(
            self.dyn_sys.x, [1.0, 0.0], decimal=5)
        np.testing.assert_array_almost_equal(
            self.dyn_sys.p, [0.1, 1.0], decimal=5)

    def test_evolve(self):
        """Test the evolution of the system over time."""
        t_range = 10.0
        self.dyn_sys.evolve(t_range)
        # Ensure that the final time matches
        self.assertAlmostEqual(self.dyn_sys.t, self.t0 + t_range, places=4)
        # Check that the solution trajectory is not empty
        self.assertGreater(self.dyn_sys.x_sol.shape[1], 0)
        # Verify that the final state has been updated
        self.assertTrue(np.allclose(self.dyn_sys.x, self.dyn_sys.x_sol[:, -1]))

    def test_plot(self):
        """Test the plot method (ensures no runtime errors)."""
        try:
            self.dyn_sys.evolve(10.0)  # Evolve the system for 1.0 time unit
            self.dyn_sys.plot([0, 1])  # Plot x and y
        except Exception as e:
            self.fail(f"Plot method raised an exception: {e}")

    def test_repr(self):
        """Test the string representation of the system."""
        repr_str = repr(self.dyn_sys)
        self.assertIn("A generic non-autonomous dynamical system", repr_str)
        self.assertIn("State vector:", repr_str)
        self.assertIn("Field vector:", repr_str)
        self.assertIn("Parameters:", repr_str)

if __name__ == "__main__":
    # unittest.main()

    parameters = {'beta': 0.1, 'omega': 1.0}
    x0 = {'x': 1.0, 'v': 0.0}

    def system(t, state_vector, p):
        x, y = state_vector
        p1, p2 = p
        xdot = y
        ydot = -p1 * y - p2 * x
        return np.array([xdot, ydot])
    
    ds = DynamicalSystem(system, 0.0, x0, parameters)
    ds.evolve(100)
    ds.plot([0,1], 'MPL')