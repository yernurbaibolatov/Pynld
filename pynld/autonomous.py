import numpy as np
from pynld.ds import DynamicalSystem
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scienceplots
from cycler import cycler

# nice colors for plotting
PLOT_COLORS = [
    '#344965', # Indigo dye
    '#FF6665', # Bittersweet
    '#1D1821', # Rich black
    '#54D6BE', # Turquoise
    '#E5AACE'  # Lavender pink
]

class AutonomousDynamicalSystem(DynamicalSystem):
    def __init__(self, system_func, x0, parameters,
                integration_params=None, jac=None):
        def wrapper_func(t, x, p):
            return system_func(x, p)
        super().__init__(wrapper_func, 0, x0, parameters, integration_params,
                        jac)
    
    def __repr__(self):
        status = "A generic autonomous dynamical system\n"
        status += f"Dimension:\t{self.N_dim}\n"
        
        status += "State vector:\n"
        for name, val in zip(self.x_names, self.x):
            status += f"\t{name}:\t{val:2.3f}\n"
        
        status += "Field vector:\n"
        for name, val in zip(self.x_names, self.xdot):
            status += f"\td{name}/dt:\t{val:2.3f}\n"
        
        status += "Parameters:\n"
        for name, val in zip(self.p_names, self.p):
            status += f"\t{name}:\t{val:2.3f}\n"

        status += "Integration parameters:\n"
        status += f"Solver: {self.solver}\n"
        status += f"N-points: {self.n_eval}\n"
        return status

    def integrate(self, t_range, tr=0):
        return super().integrate(t_range, tr)

class OneDimensional(AutonomousDynamicalSystem):
    def __init__(self, system_func, x0, parameters, integration_params=None, jac=None):
        super().__init__(system_func, x0, parameters, integration_params, jac)