import numpy as np
from pynld.ds import DynamicalSystem, IntegrationParameters
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'grid'])

PLOT_COLORS = [
    '#344966', # Indigo dye
    '#FF6666', # Bittersweet
    '#0D1821', # Rich black
    '#55D6BE', # Turquoise
    '#E6AACE'  # Lavender pink
]

def autonomous_system(t, state_vector, params):
    psi1, psi3 = state_vector
    omega, gamma, kappa, eps, Omega, L1, L3 = params
    epsc = eps/np.cos(gamma)

    # forcing
    L = L1*np.sin(psi1) + L3*np.sin(psi3)

    psi1_dot = omega - Omega + epsc*L
    psi3_dot = omega - 3*Omega + epsc*L

    return np.array([psi1_dot, psi3_dot], np.float64)

def plot_time_portraits(phs, run_time, tr_time):
    fig, axs = plt.subplots(2, 2, figsize=(14,8), layout='constrained')
    sparcity = 100
    
    phs.set_parameter('Omega', 0.30)
    phs.reset()
    # integrate using simple Runge-Kutta
    phs.integrate_local(run_time, tr_time)

    ax = axs[0,0]    
    ax.plot(phs.t_sol[::sparcity], 
            phs.x_sol[0,::sparcity], 
            lw=2.0, color=PLOT_COLORS[1])
    ax.set_ylabel(r'$\psi_1$')
    ax.set_title(r'$\varepsilon=0.4, \Omega=0.3$')

    ax = axs[1,0]
    ax.plot(phs.t_sol[::sparcity], 
            phs.x_sol[1,::sparcity], 
            lw=2.0, color=PLOT_COLORS[0])
    ax.set_xlabel(r'time')
    ax.set_ylabel(r'$\psi_3$')

    phs.set_parameter('Omega', 1.05)
    phs.reset()
    # integrate using simple Runge-Kutta
    phs.integrate_local(run_time, tr_time)
 
    ax = axs[0,1]    
    ax.plot(phs.t_sol[::sparcity], 
            phs.x_sol[0,::sparcity], 
            lw=2.0, color=PLOT_COLORS[1])
    ax.set_ylabel(r'$\psi_1$')
    ax.set_title(r'$\varepsilon=0.4, \Omega=1.05$')

    ax = axs[1,1]
    ax.plot(phs.t_sol[::sparcity], 
            phs.x_sol[1,::sparcity], 
            lw=2.0, color=PLOT_COLORS[0])
    ax.set_xlabel(r'time')
    ax.set_ylabel(r'$\psi_3$')

    plt.savefig("time_plots_autonomous.pdf", format="pdf", 
                bbox_inches="tight")
    
    plt.show()

def plot_phase_portraits(phs, run_time, tr_time):
    fig, axs = plt.subplots(1, 2, figsize=(12,6), layout='constrained')
    sparcity = 100
    
    phs.set_parameter('Omega', 0.30)
    phs.reset()
    # integrate using simple Runge-Kutta
    phs.integrate_local(run_time, tr_time)

    ax = axs[0]    
    ax.scatter(np.mod(phs.x_sol[0,::sparcity], 2*np.pi), 
               np.mod(phs.x_sol[1,::sparcity], 2*np.pi),
               s=1.0, color=PLOT_COLORS[1])
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 2*np.pi)
    ax.set_xlabel(r'$\psi_1$')
    ax.set_ylabel(r'$\psi_3$')
    ax.set_title(r'$\varepsilon=0.4, \Omega=0.3$')

    phs.set_parameter('Omega', 1.05)
    phs.reset()
    # integrate using simple Runge-Kutta
    phs.integrate_local(run_time, tr_time)

    ax = axs[1]
    ax.scatter(np.mod(phs.x_sol[0,::sparcity], 2*np.pi),
               np.mod(phs.x_sol[1,::sparcity], 2*np.pi),
               s=1.0, color=PLOT_COLORS[0])
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 2*np.pi)
    ax.set_xlabel(r'$\psi_1$')
    ax.set_ylabel(r'$\psi_3$')
    ax.set_title(r'$\varepsilon=0.4, \Omega=1.05$')

    plt.savefig("phase_plots_autonomous.pdf", format="pdf", 
                bbox_inches="tight")
    
    plt.show()

def main():
    p = {
        'omega':    1.0,
        'gamma':    0.1,
        'kappa':    -2.0,
        'eps':      0.4,
        'Omega':    0.4,
        'L1':       1.00,
        'L3':       0.10
    }
    # Set initial state and time span
    #u0 = {'x': 1.0, 'y': 0.0, 'phi1': 0.0, 'phi2': 0.0}
    u0 = {'psi1': 0.0, 'psi3': 0.0}

    ip = IntegrationParameters(solver='RK45', time_step=1e-4)
    phs = DynamicalSystem(autonomous_system, t0=0, x0=u0, parameters=p, 
                          integration_params=ip)

    #plot_time_portraits(phs, 100, 500)
    plot_phase_portraits(phs, 100, 500)

if __name__ == "__main__":
    main()