"""
Example script demonstrating the usage of Pynld.
"""

from pynld.core import DynamicalSystem
from pynld.integrators import euler_method
from pynld.plotting import plot_time_series

# Define system equations and parameters
equations = ...
parameters = ...

# Initialize the dynamical system
system = DynamicalSystem(equations, parameters)

# Set initial state and time span
initial_state = ...
time_span = ...

# Perform integration
results = euler_method(system, initial_state, time_span, step_size=0.01)

# Plot results
plot_time_series(results['time'], results['states'])