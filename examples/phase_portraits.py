import numpy as np
from pynld.ds import DynamicalSystem, IntegrationParameters
import matplotlib.pyplot as plt
import scienceplots
from scipy.integrate import odeint

plt.style.use(['science'])

PLOT_COLORS = [
    '#344966', # Indigo dye
    '#FF6666', # Bittersweet
    '#0D1821', # Rich black
    '#55D6BE', # Turquoise
    '#E6AACE'  # Lavender pink
]


def system(state_vector, t):
    x, y = state_vector
    dot_x = y + x - x**3
    dot_y = -y

    return [dot_x, dot_y]


def main():
    # Grid setup
    x_min, x_max, x_pts = -5, 5, 30
    y_min, y_max, y_pts = -5, 5, 30 
    x = np.linspace(x_min, x_max, x_pts)
    y = np.linspace(y_min, y_max, y_pts)
    X, Y = np.meshgrid(x, y)

    # Compute derivatives
    U = Y + X - X**3
    V = -Y

    # Normalize
    M = np.hypot(U, V)
    U /= M
    V /= M

    # Plotting
    plt.figure(figsize=(8,8))
    plt.quiver(X, Y, U, V, color=PLOT_COLORS[0], alpha=0.6)

    # Trajectories
    initial_conditions = [(2.1, 2.1), (-2, -2.5)]
    t = np.linspace(0, 15, 1000)

    for ic in initial_conditions:
        sol = odeint(system, ic, t)
        plt.plot(sol[:, 0], sol[:, 1], c=PLOT_COLORS[1])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Phase Portrait')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

if __name__=="__main__":
    main()

