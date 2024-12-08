"""
Unit tests for core functionalities.
"""

import numpy as np
from pynld.ds import DynamicalSystem 


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
    ds.integrate(100, 10)
    ds.time_plot()