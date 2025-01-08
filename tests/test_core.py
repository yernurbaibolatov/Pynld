"""
Unit tests for core functionalities.
"""

import numpy as np
from pynld.ds import DynamicalSystem 
from pynld.autonomous import AutonomousDynamicalSystem, OneDimensional


if __name__ == "__main__":
    # unittest.main()

    parameters = {'beta': 0.1, 'omega': 1.0}
    x0 = {'x': 1.0, 'v': 0.0}

    p1 = {'nu': 1.0, 'eps': 0.2, 'gamma': 0.1}
    x01 = {'psi': 0.0}

    def system(state_vector, p):
        x, y = state_vector
        p1, p2 = p
        xdot = y
        ydot = -p1 * y - p2 * x
        return np.array([xdot, ydot])
    
    def system1D(x, p):
        psi = x
        nu, eps, gamma = p

        dot_psi = nu + eps*np.sin(psi + gamma)

        return np.array([dot_psi])

    
    #ds = AutonomousDynamicalSystem(system, x0, parameters)
    #ds = DynamicalSystem(system, 0, x0, parameters)
    #ds = AutonomousDynamicalSystem(system1D, x01, p1)
    ds = OneDimensional(system1D, x01, p1)
    ds.integrate(100, 10)
    ds.time_plot()
    print(ds)