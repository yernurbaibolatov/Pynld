"""
Core functionalities for dynamical systems.
"""
import numpy as np
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed, cpu_count
from pynld.abstract_integrator import AbstractIntegrator
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler
import scienceplots

# nice colors for plotting
PLOT_COLORS = [
    '#344965', # Indigo dye
    '#FF6665', # Bittersweet
    '#1D1821', # Rich black
    '#54D6BE', # Turquoise
    '#E5AACE'  # Lavender pink
]


class IntegrationParameters:
    def __init__(self, solver='LSODA', time_step = 1e-3, 
                 accuracy = 1e-5, n_eval = 50_000):
        self.solver = solver
        self.time_step = time_step
        self.accuracy = accuracy
        self.n_eval = n_eval

class DynamicalSystem(AbstractIntegrator):
    def __init__(self, system_func, t0, x0, parameters, 
                 integration_params = None, jac=None):
        """
        Initialize the dynamical system.

        Parameters:
        - system_func: callable, system of equations (dx/dt = f(t, x, p)).
        - t0: float, initial time.
        - x0: dict, initial conditions of the system.
        - parameters: dict, parameters of the system.
        - integration_params: IntegrationParameters object, optional.
        - jac: callable, Jacobian of the system, optional.
        """
        if not callable(system_func):
            raise TypeError("The 'system' argument must be a callable.")
        if not isinstance(x0, dict) or not isinstance(parameters, dict):
            raise TypeError("'x0' and 'parameters' must be dictionaries.")
        
        super().__init__(
            solver=integration_params.solver if integration_params else 'LSODA',
            time_step=integration_params.time_step if integration_params else 1e-3,
            n_eval=integration_params.n_eval if integration_params else 50_000,
        )

        self.system_func = system_func
        self.jac = jac

        # extracting the parameters
        self.p_names = list(parameters.keys())
        self.p = np.array(list(parameters.values()), dtype=np.float64)

        # extracting the initial conditions
        self.x_names = list(x0.keys())
        self.x = np.array(list(x0.values()), dtype=np.float64)

        self.t = t0
        self.xdot = system_func(self.t, self.x, self.p)

        # save the initial conditions
        self.initial_t = t0
        self.initial_x = self.x

        # technical parameters
        self.N_dim = len(x0) # dimension of the system (without time)

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
        status += f"Solver: {self.solver}\n"
        status += f"N-points: {self.n_eval}\n"
        return status
    
    def __plot_init__(self, notebook=False):
        # initialize the matplotlib parameters
        if notebook:
            plt.style.use(['science', 'nature', 'notebook'])
        else:
            plt.style.use(['science', 'nature'])

        plt.rcParams.update({
            # Figure and layout
            'figure.figsize': [12, 6],                 # Default figure size
            'axes.prop_cycle': cycler(color=PLOT_COLORS),  # Custom color cycle for lines
            'lines.linewidth': 2.0,                   # Line width
            'lines.markersize': 8,                    # Marker size
            # Grid
            'axes.grid': True,                        # Enable grid
            'grid.alpha': 0.7,                        # Transparency of grid lines
            'grid.linestyle': '--',                   # Dashed grid lines
            'grid.linewidth': 0.6,                    # Grid line width

            # Colormap (for plots like heatmaps)
            'image.cmap': 'viridis',                  # Default colormap
            'image.interpolation': 'nearest',         # No smoothing in heatmaps
        })

    def set_parameter(self, name, val):
        # Set the parameter 'name' to value 'val'
        if name in self.p_names:
            i = self.p_names.index(name)
            self.p[i] = val
        else:
            raise ValueError(f"{name} is not found in the list of parameters.")

    def system(self, t, x, p):
        return self.system_func(t, x, p)

    def integrate(self, t_range, tr=0):
        # Evolves the system by t_range
        # tr is the transient time in the beginning
        tr_span = [self.t, self.t + tr]
        if self.jac is None:
           tr_sol = solve_ivp(self.system, 
                           t_span=tr_span, 
                           y0=self.x, 
                           args=(self.p,), 
                           method=self.solver)
        else: 
            tr_sol = solve_ivp(self.system, 
                           t_span=tr_span, 
                           y0=self.x, 
                           args=(self.p,), 
                           method=self.solver,
                           jac=self.jac)
        self.t = tr_sol.t[-1]
        self.x = tr_sol.y[:,-1]

        # actual solution
        t_span = [self.t, self.t + t_range]
        t_eval = np.linspace(self.t, self.t + t_range, 
                           self.n_eval)
        # integrate the system
        if self.jac is None:
            sol = solve_ivp(self.system, 
                        t_span=t_span, 
                        y0=self.x, 
                        t_eval=t_eval, 
                        args=(self.p,),
                        method=self.solver)
        else:
            sol = solve_ivp(self.system, 
                        t_span=t_span, 
                        y0=self.x, 
                        t_eval=t_eval, 
                        args=(self.p,),
                        method=self.solver,
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
        self.reset()
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

    def run_parameter(self, eval_f, p, p_range, t_range, tr=0, parallel=-1):
        """
        Calls the `evaluate` function for each value of parameter `p` in
        `p_range`.
        Returns an array of outputs of `evaluate` for each `p`.
        Parameters:
            eval_f: a callable that returns a single or 
            array of values.
            t_range and tr: parameters that are passed
            to the integrate method.
            p: name of the parameter that should be considered
            p_range: range of change of parameter p
            parallel: number of CPU cores to use for parallel computation.
            If 0, then single core is used, if -1 (default), 
            all available CPU cores are used.
        """
        if p not in self.p_names:
            raise ValueError(f"{p} is not found in the list of parameters.") 

        def run(p_val):
            self.set_parameter(p, p_val)

            return self.evaluate(eval_f, t_range, tr)
        
        print(f"Simulation is running for parameter '{p}' in range: [{p_range[0]}:{p_range[-1]}]")
        num_cores = "all cores" if parallel == -1 else str(parallel)
        print(f"Using {num_cores} for parallel computing ({cpu_count()} available cores)")
        print("...")
        run_vals = Parallel(n_jobs=parallel)(delayed(run)(p_val) 
                                             for p_val in p_range)
        
        print("Simulation finished.")
        return np.asarray(run_vals)

    def time_plot(self, notebook=False):
        self.__plot_init__(notebook)

        for i in range(len(self.x_sol)):
            plt.plot(self.t_sol, self.x_sol[i], label=f"{self.x_names[i]}")

        plt.xlabel('Time')
        plt.ylabel('Variables')
        plt.legend(loc='best')
        plt.show()

   