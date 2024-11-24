import matplotlib.pyplot as plt
import scienceplots
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool

PLOT_COLORS = [
    '#344966', # Indigo dye
    '#FF6666', # Bittersweet
    '#0D1821', # Rich black
    '#55D6BE', # Turquoise
    '#E6AACE'  # Lavender pink
]

# Time evolution plots
def evolution_plot(ds, var_names=None, method='Bokeh', notebook=True):
    """
    Plot the time evolution of the latest solution.
    If method == 'Bokeh' (default), uses the Bokeh background.
    If method == 'MPL', uses Matplotlib.
    If notebook == True (default), then adjusts the plot for Jupyter nb.
    """
    if ds.x_sol is None or ds.t_sol is None:
        raise ValueError("No solution found. Please evolve the system first.")

    if var_names is None:
        var_names = ds.x_names

    if method=='Bokeh':
        evolution_plot_bokeh(ds, var_names, notebook)
    elif method=='MPL':
        evolution_plot_mpl(ds, var_names, notebook)
    else:
        raise ValueError("Use 'Bokeh' or 'MPL'")

def evolution_plot_bokeh(ds, var_names, notebook):
    p = figure(title="Time evolution", 
               x_axis_label="time",
               y_axis_label="variables",
               width=1200,
               height=600)
    if notebook:
        output_notebook()

    indices = [ds.x_names.index(var) for var in var_names if var in ds.x_names]
    for i, var in zip(indices,var_names):
        p.line(ds.t_sol, ds.x_sol[i],
               legend_label=f"{var}",
               line_width=2, color=PLOT_COLORS[i])
        
    p.add_tools(HoverTool(tooltips=[("Time", "@x"), ("Value", "@y")]))
    show(p)

def evolution_plot_mpl(ds, var_names, notebook):
    if notebook:
        plt.style.use(['science', 'grid', 'notebook'])
    else:
        plt.style.use(['science', 'grid'])

    plt.figure(figsize=(12,6))
    indices = [ds.x_names.index(var) for var in var_names if var in ds.x_names]
    for i, var in zip(indices, var_names):
        plt.plot(ds.t_sol, ds.x_sol[i], '-', lw=2.0, 
                color=PLOT_COLORS[i], label=f"{var}")
    plt.xlabel("time")
    plt.ylabel("variables")
    plt.title("Time evolution")
    plt.legend()
    plt.show()

# Phase portrait plots
