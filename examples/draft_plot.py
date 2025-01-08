import numpy as np
from scipy.optimize import root_scalar
from pynld.ds import DynamicalSystem
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

def system(t, state_vector, params):
    x, y, phi1, phi2 = state_vector
    omega, gamma, kappa, eps, Omega, L1, L2= params
    eta = -kappa/2
    beta = np.tan(gamma)
    nu = omega + eta*beta
    epsc = eps/np.cos(gamma)
    Wt = Omega*t

    # forcing terms
    Fx = L1*np.cos(Wt) + L2*np.cos(3*Wt)
    Fy = L1*np.sin(Wt) + L2*np.sin(3*Wt)

    f1 = L1*np.sin(1*Wt-phi1-gamma)
    f2 = L2*np.sin(3*Wt-phi1-gamma)

    s1 = L1*np.sin(1*Wt-phi2-gamma)
    s2 = L2*np.sin(3*Wt-phi2-gamma)

    d1 = np.atan((omega-1*Omega)/kappa)
    d2 = np.atan((omega-3*Omega)/kappa)

    s11 = L1*L1*np.cos(d1)*(np.sin(-0*Wt+d1)+np.sin(2*(1*Wt-phi2)-d1))
    s12 = L1*L2*np.cos(d2)*(np.sin(-2*Wt+d2)+np.sin(2*(2*Wt-phi2)-d2))
    s21 = L2*L1*np.cos(d1)*(np.sin(+2*Wt+d1)+np.sin(2*(2*Wt-phi2)-d1))
    s22 = L2*L2*np.cos(d2)*(np.sin(+0*Wt+d2)+np.sin(2*(3*Wt-phi2)-d2))
    # cartesian equations
    xdot = eta*x - nu*y - eta*(x - beta*y)*(x**2 + y**2) + eps*Fx
    ydot = eta*y + nu*x - eta*(y + beta*x)*(x**2 + y**2) + eps*Fy

    # first-order phase equation
    phi_dot_1 = omega + epsc*(f1+f2)

    # second-order phase equation
    phi_dot_2 = omega + epsc*(s1+s2) + epsc**2*(s11+s12+s21+s22)/(2*kappa)

    return np.array([xdot, ydot, phi_dot_1, phi_dot_2], np.float64)

def frequency(t, state_vector, state_vector_dot):
    x, y, _, _ = state_vector
    xdot, ydot, phi_dot_1, phi_dot_2 = state_vector_dot
    freq = (ydot*x - xdot*y)/(x**2 + y**2)
    return np.array([freq, phi_dot_1, phi_dot_2])

def plot_phase_portraits(phs, p, run_time, tr_time):
    plt.figure(figsize=(8,8))
    
    phs.set_parameter('eps', 0.5)
    phs.evaluate(frequency, run_time, tr_time)

    plt.subplot(221)
    plt.plot(phs.x_sol[0], phs.x_sol[1], lw=1.0, color=PLOT_COLORS[1])
    plt.ylabel(r'$y$')
    plt.title(r"$\varepsilon=0.5, \Omega=0.4$")

    phs.set_parameter('eps', 0.6)
    phs.evaluate(frequency, run_time, tr_time)

    plt.subplot(222)
    plt.plot(phs.x_sol[0], phs.x_sol[1], lw=1.0, color=PLOT_COLORS[1])
    plt.title(r"$\varepsilon=0.6, \Omega=0.4$")

    phs.set_parameter('eps', 0.7)
    phs.evaluate(frequency, run_time, tr_time)

    plt.subplot(223)
    plt.plot(phs.x_sol[0], phs.x_sol[1], lw=1.0, color=PLOT_COLORS[1])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r"$\varepsilon=0.7, \Omega=0.4$")

    phs.set_parameter('eps', 0.7)
    phs.evaluate(frequency, run_time, tr_time)

    plt.subplot(224)
    plt.plot(phs.x_sol[0], phs.x_sol[1], lw=1.0, color=PLOT_COLORS[1])
    plt.xlabel(r'$x$')
    plt.title(r"$\varepsilon=0.8, \Omega=0.4$")

    #plt.savefig("draft_plots/phase_portraits_2.pdf", format="pdf", 
    #            bbox_inches="tight")
    
    plt.show()

def plot_tongues(phs, p, run_time, tr_time, Omega_range):
    freqs = phs.run_parameter(frequency, 'Omega', Omega_range, 
                              t_range=run_time, tr=tr_time)

    freq_xy = freqs[:,0]
    freq_f1 = freqs[:,1]
    freq_f2 = freqs[:,2]

    # calculate the analytic boundaries
    eps = p['eps']/np.cos(p['gamma'])
    w = p['omega']
    L1, L2 = p['L1'], p['L2']
    h = 0.5*(eps*L1/p['kappa'])**2
    W_min_3 = (w - eps*L2)/3
    W_max_3 = (w + eps*L2)/3
    W_min_3s = (w*(1+h) - eps*L2)/(3+h)
    W_max_3s = (w*(1+h) + eps*L2)/(3+h)
    W_min_1 = (w - eps*L1)/1
    W_max_1 = (w + eps*L1)/1

    # plot the results
    plt.figure(figsize=(12,6))
    plt.plot(Omega_range, freq_xy-Omega_range, lw=1.0, color=PLOT_COLORS[0],
             label="Full system")
    plt.plot(Omega_range, freq_f1-Omega_range, lw=1.0, color=PLOT_COLORS[1],
             label="First-order")
    plt.plot(Omega_range, freq_f2-Omega_range, lw=1.0, color=PLOT_COLORS[3],
             label="Second-order")

    plt.xlabel(r"$\Omega$")
    plt.ylabel(r"$\dot{\phi}$/$\Omega$")
    plt.title(rf"$\varepsilon={p['eps']}$")
    plt.legend()

    #plt.axvline(W_min_3, color='red')
    #plt.axvline(W_max_3, color='red')
    #plt.axvline(W_min_3s, color='grey')
    #plt.axvline(W_max_3s, color='grey')
    #plt.axvline(W_min_1, color='red')
    #plt.axvline(W_max_1, color='red')

    #plt.savefig(f"draft_plots/domain_zoom_eps={p['eps']}.pdf", format="pdf", 
    #            bbox_inches="tight")
    #"""
    
    plt.show()

def plot_stability(p):
    omega, gamma, kappa, eps, Omega, L1, L2 = p
    eta = -kappa/2
    beta = np.tan(gamma)
    nu = omega + eta*beta
    epsc = eps/np.cos(gamma)

    def slow_phase(psi):
        return omega - 3*Omega + epsc*np.sin(psi + gamma)
    sol = []
    interval = np.linspace(0, 2*np.pi, 10)

    for i in range(len(interval)-1):
        a, b = interval[i], interval[i+1]
        if slow_phase(a)*slow_phase(b) < 0:
            sol_cur = root_scalar(slow_phase, 
                                  bracket=[a, b], method='brentq')
            sol.append(sol_cur)
    nulls = np.zeros(len(sol))

    psi = np.linspace(0, 2*np.pi, 1000)

    plt.figure(figsize=(12,6))
    plt.plot(psi, slow_phase(psi), 
             lw=2.0, color=PLOT_COLORS[0])
    
    plt.axhline(0, color='red')
    plt.scatter(nulls, sol)

    plt.show()

def main():
    # parameters of the simulation
    p = {
    'omega':    1.0,
    'gamma':    0.1,
    'kappa':    -2.0,
    'eps':      0.3,
    'Omega':    0.3,
    'L1':       1.00,
    'L2':       0.00
    }

    # set initial state and time span
    u0 = {'x': 1.0, 'y': 0.0, 'phi1': 0.0, 'phi2': 0.0}

    # dynamical system
    phs = DynamicalSystem(system, t0=0, x0=u0, parameters=p)
    
    # simulation parameters
    Omega_range = np.linspace(0.20, 1.20, 1000)
    run_time = 10_000
    tr_time = 100

    # * Phase portraits
    #plot_phase_portraits(phs, p, run_time, tr_time)

    # * Run the simulation
    #plot_tongues(phs, p, run_time, tr_time, Omega_range)

    # * Stability analysis
    plot_stability(p.values())


if __name__ == "__main__":
    main()