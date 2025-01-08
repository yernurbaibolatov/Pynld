import numpy as np
from pynld.ds import DynamicalSystem
import matplotlib.pyplot as plt


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
    def __init__(self, system_func, x0, parameters, integration_params=None,
                 jac=None):
        if len(x0)>1:
            raise ValueError("Initial conditions must be one-dimensional")
        x0_val = next(iter(x0.values()))
        if len(system_func(x0_val, parameters.values())) > 1:
            raise ValueError(f"{system_func} must return a one-dimnensional object")
        
        super().__init__(system_func, x0, parameters, integration_params, jac)

    def __repr__(self):
        status = "A one-dimensional dynamical system\n"
        
        status += "State variable:\n"
        for name, val in zip(self.x_names, self.x):
            status += f"\t{name}:\t{val:2.3f}\n"
        
        status += "State variable time-derivative:\n"
        for name, val in zip(self.x_names, self.xdot):
            status += f"\td{name}/dt:\t{val:2.3f}\n"
        
        status += "Parameters:\n"
        for name, val in zip(self.p_names, self.p):
            status += f"\t{name}:\t{val:2.3f}\n"

        status += "Integration parameters:\n"
        status += f"Solver: {self.solver}\n"
        status += f"N-points: {self.n_eval}\n"
        return status

    def phase_protrait(self, notebook):
        self.__plot_init__(notebook)

        