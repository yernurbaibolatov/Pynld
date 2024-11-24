"""
Core functionalities for dynamical systems.
"""
import numpy as np
from integrators import runge_kutta_stepper

class IntegrationParameters:
    def __init__(self, solver='RK45', time_step = 1e-2, accuracy = 1e-5):
        self.solver = solver
        self.time_step = time_step
        self.accuracy = accuracy

class DynamicalSystem:
    def __init__(self, system, t0, x0, parameters, integration_params = None):
        """
        Initialize the dynamical system.
        
        Parameters:
            equations (function): A function defining the system of equations (dx/dt = f(t, x, p)). x and p are numpy arrays 
            t0: Initial time
            x0: A dictionary of initial conditions for the system.
            parameters: A dictionary of parameters of the system
        Example:
            parameters = {
                'nu':   1.1,
                'eta':  0.1,
                'beta': 1.0
            }
            x0 = {
                'x':    1.0,
                'y':    0.0
            }  
        """
        self.system = system
        self.integration_params = integration_params or IntegrationParameters()

        # extracting the parameters
        self.p_names = list(parameters.keys())
        self.p = np.array(list(parameters.values()), dtype=np.float64)

        # extracting the initial conditions
        self.x_names = list(x0.keys())
        self.x = np.array(list(x0.values()), dtype=np.float64)

        self.t = t0
        self.xdot = system(self.t, self.x, self.p)

        # technical parameters
        self.N_dim = len(x0) # dimension of the system (without time)
        self.dt = 1e-4 # time step for solvers
        self.solver = 'RK45' # default solver is Runge-Kutta 4(5)

        # trajectory of the last integration
        self.t_sol = np.zeros(0, dtype=np.float64)
        self.x_sol = np.zeros((self.N_dim,0), dtype=np.float64)
        self.xdot_sol = np.zeros((self.N_dim,0), dtype=np.float64)

    def __repr__(self):
        status = f"A generic non-autonomous dynamical system\n"
        status += f"Dimension:\t{self.N_dim + 1}\n"
        
        status += f"State vector:\n"
        status += f"\tt:\t{self.t:2.3f}\n"
        for name, val in zip(self.x_names, self.x):
            status += f"\t{name}:\t{val:2.3f}\n"
        
        status += f"Field vector:\n"
        status += f"\tdt/dt:\t1\n"
        for name, val in zip(self.x_names, self.xdot):
            status += f"\td{name}/dt:\t{val:2.3f}\n"
        
        status += f"Parameters\n"
        for name, val in zip(self.p_names, self.p):
            status += f"\t{name}:\t{val:2.3f}\n"

        return status
    
    def evolve(self, t_range):
        # Evolves the system by t_range
        self.t_sol = np.arange(self.t, self.t + t_range, 
                               self.integration_params.time_step)
        self.x_sol = np.zeros((self.N_dim, len(self.t_sol)), dtype=np.float64)
        self.xdot_sol = np.zeros_like(self.x_sol)
        self.x_sol[:,0] = self.x
        self.xdot_sol[:,0] = self.xdot

        for i in range(len(self.t_sol)-1):
            self.x_sol[:,i+1] = runge_kutta_stepper(self.system, 
                                                    self.t_sol[i], 
                                                    self.x_sol[:,i], self.p, 
                                                    self.integration_params.time_step)
            self.xdot_sol[:,i+1] = self.system(self.t_sol[i+1],
                                               self.x_sol[:,i+1],
                                               self.p)
        
        self.t = self.t_sol[-1]
        self.x = self.x_sol[:,-1]
        self.xdot = self.xdot_sol[:,-1]

            
