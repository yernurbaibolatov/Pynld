from abc import ABC, abstractmethod
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

class AbstractPlotter(ABC):
    def __init__(self):
        rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'font.family': 'serif',
            'mathtext.fontset': 'stix',
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.8,
            'lines.linewidth': 1.5,
            'figure.figsize': [6.4, 4.8],
        })
        plt.rc('axes', cycler(color=PLOT_COLORS))
    