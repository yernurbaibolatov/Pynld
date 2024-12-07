import numpy as np
from core import *
import matplotlib.pyplot as plt
from cycler import cycler
from configs import PLOT_COLORS

print(PLOT_COLORS)

from matplotlib import rcParams

# Update settings
rcParams.update({
    'figure.dpi': 300,                   # High resolution
    'savefig.dpi': 300,                  # High resolution for saved figures
    'font.size': 10,                     # Base font size
    'axes.titlesize': 10,                # Font size for axis titles
    'axes.labelsize': 8,                # Font size for axis labels
    'xtick.labelsize': 8,               # Font size for x-axis tick labels
    'ytick.labelsize': 8,               # Font size for y-axis tick labels
    'legend.fontsize': 6,               # Font size for legends
    'lines.linewidth': 1.5,              # Line width
    'lines.markersize': 6,               # Marker size
    'axes.linewidth': 1.0,               # Axis border width
    'grid.linewidth': 0.5,               # Grid line width
    'grid.alpha': 0.8,                   # Grid line transparency
    'legend.loc': 'best',                # Optimal legend placement
    'mathtext.fontset': 'stix',          # Use STIX fonts for math
    'font.family': 'serif',              # Use serif fonts
    'figure.figsize': [6.0, 3.0],        # Default figure size
    'axes.grid': True,                   # Enable grid by default
})

#rcParams['text.usetex'] = True
#rcParams['font.family'] = 'serif'
plt.rc('axes', prop_cycle=cycler('color', PLOT_COLORS))

# Data
x = np.linspace(0, 10, 100)
y1 = np.sin(x) * np.exp(-x / 5)
y2 = np.sin(1.2*x) * np.exp(-x / 5)
y3 = np.sin(2.0*x) * np.exp(-x / 5)
y4 = np.sin(x) * np.exp(-x / 3)
y5 = np.sin(1.5*x) * np.exp(-x / 3)

# Plot
plt.plot(x, y1, label=r'$\sin(x) \cdot e^{-x/5}$')
plt.plot(x, y2, label=r'$\sin(1.2x) \cdot e^{-x/5}$')
plt.plot(x, y3, label=r'$\sin(2x) \cdot e^{-x/5}$')
plt.plot(x, y4, label=r'$\sin(x) \cdot e^{-x/3}$')
plt.plot(x, y5, label=r'$\sin(1.5x) \cdot e^{-x/3}$')
plt.title('Damped Oscillation')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(frameon=True, framealpha=0.8, loc='best')
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.8)
plt.tight_layout()
plt.axis('equal') 

plt.show()


