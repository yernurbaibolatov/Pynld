"""
Example script demonstrating the usage of Pynld.
"""

from pynld.core import DynamicalSystem
from pynld.plotting import evolution_dot_plot
import numpy as np

# Define system equations and parameters
def system(t, state_vector, params):
    phi = state_vector
    omega, gamma, Omega, eps, A1, A2, A3 = params
    f1 = A1*np.sin(1*Omega*t-phi-gamma)
    f2 = A2*np.sin(3*Omega*t-phi-gamma)
    f3 = A3*np.sin(5*Omega*t-phi-gamma)
    phi_dot = omega + (eps/np.cos(gamma))*(f1+f2+f3)
    return np.array([phi_dot])

def system_jac(t, state_vector, params):
    phi = state_vector
    omega, gamma, Omega, eps, A1, A2, A3 = params
    f1 = -A1*np.cos(1*Omega*t-phi-gamma)
    f2 = -A2*np.cos(3*Omega*t-phi-gamma)
    f3 = -A3*np.cos(5*Omega*t-phi-gamma)

    dfdphi = (eps/np.cos(gamma))*(f1+f2+f3)
    return np.array([dfdphi])

parameters = {
    'omega':    1.0,
    'gamma':    0.1,
    'Omega':    0.4,
    'eps':      0.3,
    'A1':       1.00,
    'A2':       0.10,
    'A3':       0.01
}

# Set initial state and time span
initial_state = {'phi': 0.0}

# Initialize the dynamical system
phs = DynamicalSystem(system, t0=0, x0=initial_state, 
                      parameters=parameters, jac=system_jac)

phs.integrate(1000, 10)