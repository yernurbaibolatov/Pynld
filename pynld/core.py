"""
Core functionalities for dynamical systems.
"""
import numpy as np
from scipy.integrate import solve_ivp

class IntegrationParameters:
    def __init__(self, solver='LSODA', time_step = 1e-3, 
                 accuracy = 1e-5, n_eval = 50_000):
        self.solver = solver
        self.time_step = time_step
        self.accuracy = accuracy
        self.n_eval = n_eval

class DynamicalSystem:
    def __init__(self, system, t0, x0, parameters, 
                 integration_params = None, jac=None):
        """
        Initialize the dynamical system.
        
        Parameters:
            equations (function): A function defining the system of equations (dx/dt = f(t, x, p)). x and p are numpy arrays 
            t0: Initial time
            x0: A dictionary of initial conditions for the system.
            parameters: A dictionary of parameters of the system
            jac (optional): the Jacobian of the system 
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
        if not callable(system):
            raise TypeError("The 'system' argument must be a callable.")
        if not isinstance(x0, dict) or not isinstance(parameters, dict):
            raise TypeError("'x0' and 'parameters' must be dictionaries.")
        

        self.system = system
        self.jac = jac
        self.integration_params = integration_params or IntegrationParameters()

        # extracting the parameters
        self.p_names = list(parameters.keys())
        self.p = np.array(list(parameters.values()), dtype=np.float64)

        # extracting the initial conditions
        self.x_names = list(x0.keys())
        self.x = np.array(list(x0.values()), dtype=np.float64)

        self.t = t0
        self.xdot = system(self.t, self.x, self.p)

        # save the initial conditions
        self.initial_t = t0
        self.initial_x = self.x

        # technical parameters
        self.N_dim = len(x0) # dimension of the system (without time)
        self.dt = self.integration_params.time_step # time step for solvers
        # default solver is LSODA
        self.solver = self.integration_params.solver 

        # trajectory of the last integration
        self.t_sol = np.zeros(0, dtype=np.float64)
        self.x_sol = np.zeros((self.N_dim,0), dtype=np.float64)
        self.xdot_sol = np.zeros((self.N_dim,0), dtype=np.float64)
        self.n_points = 0 # number of point in the solution
        # array of evaluated values at point of the solution
        self.f_sol = np.zeros(0, dtype=np.float64) 

    def __repr__(self):
        status = "A generic non-autonomous dynamical system\n"
        status += f"Dimension:\t{self.N_dim + 1}\n"
        
        status += "State vector:\n"
        status += f"\tt:\t{self.t:2.3f}\n"
        for name, val in zip(self.x_names, self.x):
            status += f"\t{name}:\t{val:2.3f}\n"
        
        status += "Field vector:\n"
        status += "\tdt/dt:\t1\n"
        for name, val in zip(self.x_names, self.xdot):
            status += f"\td{name}/dt:\t{val:2.3f}\n"
        
        status += "Parameters:\n"
        for name, val in zip(self.p_names, self.p):
            status += f"\t{name}:\t{val:2.3f}\n"

        status += "Integration parameters:\n"
        status += f"Solver: {self.integration_params.solver}\n"
        status += f"Time step: {self.integration_params.time_step}\n"
        return status
    
    def set_parameter(self, name, val):
        # Set the parameter 'name' to value 'val'
        if name in self.p_names:
            i = self.p_names.index(name)
            self.p[i] = val
        else:
            raise ValueError(f"{name} is not found in the list of parameters.")

    def integrate(self, t_range, tr=0):
        # Evolves the system by t_range
        # tr is the transient time in the beginning
        tr_span = [self.t, self.t + tr]
        if self.jac is None:
           tr_sol = solve_ivp(self.system, 
                           t_span=tr_span, 
                           y0=self.x, 
                           args=(self.p,), 
                           method=self.integration_params.solver)
        else: 
            tr_sol = solve_ivp(self.system, 
                           t_span=tr_span, 
                           y0=self.x, 
                           args=(self.p,), 
                           method=self.integration_params.solver,
                           jac=self.jac)
        self.t = tr_sol.t[-1]
        self.x = tr_sol.y[:,-1]

        # actual solution
        t_span = [self.t, self.t + t_range]
        t_eval = np.linspace(self.t, self.t + t_range, 
                           self.integration_params.n_eval)
        # integrate the system
        if self.jac is None:
            sol = solve_ivp(self.system, 
                        t_span=t_span, 
                        y0=self.x, 
                        t_eval=t_eval, 
                        args=(self.p,),
                        method=self.integration_params.solver)
        else:
            sol = solve_ivp(self.system, 
                        t_span=t_span, 
                        y0=self.x, 
                        t_eval=t_eval, 
                        args=(self.p,),
                        method=self.integration_params.solver,
                        jac=self.jac)
        # store the solution
        self.t_sol = sol.t
        self.x_sol = sol.y.copy()
        self.n_points = len(self.t_sol)
        self.xdot_sol = np.zeros_like(self.x_sol)
        for i in range(self.n_points):
            self.xdot_sol[:,i] = self.system(self.t_sol[i],
                                             self.x_sol[:,i],
                                             self.p)
        
        # update the system state
        self.t = self.t_sol[-1]
        self.x = self.x_sol[:,-1]
        self.xdot = self.xdot_sol[:,-1]
        return

    def evaluate(self, eval_f, t_range, tr=0, reduce="average"):
        """
        Evaluate a function eval_f(t, x, xdot) for
        each point of the solution obtained from the
        integrate method. Returns a vector of computed
        values.
        Parameters:
            eval_f: a callable that returns a single or 
            array of values.
            t_range and tr: parameters that are passed
            to the integrate method.
        """
        self.integrate(t_range, tr)
        # check the dimensions of the eval_f function
        N_dim = len(eval_f(self.t_sol[0], self.x_sol[:,0], self.xdot_sol[:,0]))
 
        f_eval = np.zeros((N_dim, self.n_points), dtype=np.float64)
        for i in range(self.n_points):
            f_eval[:,i] = eval_f(self.t_sol[i],
                                 self.x_sol[:,i],
                                 self.xdot_sol[:,i])
        
        # reduce = average
        return np.mean(f_eval, axis=1)
    
    def reset(self):
        # Resets the system back to its initial condition
        self.t = self.initial_t
        self.x = self.initial_x
        self.xdot = self.system(self.t, self.x, self.p)

        self.t_sol = np.zeros(0, dtype=np.float64)
        self.x_sol = np.zeros((self.N_dim,0), dtype=np.float64)
        self.xdot_sol = np.zeros((self.N_dim,0), dtype=np.float64)
        self.n_points = 0 # number of point in the solution