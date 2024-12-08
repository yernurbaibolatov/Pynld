import numpy as np
from pynld.ds import DynamicalSystem 
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scienceplots
from cycler import cycler

plt.style.use(['science','grid','nature'])
plt.rc('axes', prop_cycle=cycler('color', PLOT_COLORS))

class OneDimensional(AutonomousDynamicalSystem):
    def __init__(self, system, x0, parameters, integration_params=None):
        if len(x0)>1:
            raise ValueError(f"Initial conditions {x0} must be one-dimensional")

        if len(system(x0, parameters))>1:
            raise ValueError(f"The {system} must return a one-dimensional object")

        super().__init__(system, 0, x0, parameters, 
                         integration_params, jac=None)
    
    def __repr__(self):
        status = "One-dimensional dynamical system\n"
        
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
        status += f"Solver: {self.integration_params.solver}\n"
        status += f"Time step: {self.integration_params.time_step}\n"
        return status
    
    