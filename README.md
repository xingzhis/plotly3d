# plotly3d
A helper for ploting scatter plots and trajectory plots with plotly.

[Krishnaswamy Lab](https://krishnaswamylab.org/members), Yale University

[![Latest PyPI version](https://img.shields.io/pypi/v/plotly3d.svg)](https://pypi.org/project/plotly3d/)
[![PyPI license](https://img.shields.io/pypi/l/plotly3d.svg)](https://pypi.python.org/pypi/plotly3d/)
[![PyPI download month](https://img.shields.io/pypi/dm/plotly3d.svg)](https://pypi.python.org/pypi/plotly3d/)
[![PyPI download day](https://img.shields.io/pypi/dd/plotly3d.svg)](https://pypi.python.org/pypi/plotly3d/)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)


## Install

The package can be directly installed via pip:
```
pip install plotly3d
```

## APIs

<details><summary>
Scatter Plots
</summary>

```
def scatter(points, colors=None, **kwargs):
    """
    Plots 3D scatter plot with optional rescaling, coloring, and customization.

    Parameters:
    - points: Array of points to plot.
    - colors: Optional array of colors for each point.
    - scaler: Optional scaler object to rescale points. If None and rescaling is enabled, MinMaxScaler is used.
    - **kwargs: Additional optional arguments:
        - s (float): Size of the markers.
        - alpha (float): Opacity of the markers.
        - force_continuous (bool): Force treating colors as continuous even if they seem categorical.
        - title (str): Title of the plot.
        - filename (str): If provided, saves the plot to this file.
        - rescale (bool): If True, rescales points using the provided or default scaler.
        - fig (go.Figure): Plotly figure object to which the scatter plot will be added. If None, a new figure is created.
        - xtitle (str), ytitle (str), ztitle (str): Titles for the X, Y, and Z axes.

    Returns:
    - Plotly figure containing the scatter points plotted in 3D space.
    """
```

</details>

<details><summary>
Trajectory Plots
</summary>

```
def trajectories(trajs, colors=None, **kwargs):
    """
    Plots trajectories in 3D space using Plotly, with unique colors for each category and a single legend entry per category.

    Parameters:
    - trajs: numpy array of shape (a, b, c), where
      a: time bins, b: trajectory index, c: space dimension (c=3 for 3D).

    Returns:
    - Plotly figure containing the trajectories plotted in 3D space.
    """
```

</details>