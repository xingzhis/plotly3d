# plotly3d
A helper for ploting scatter plots and trajectory plots with plotly.

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
    """
```

</details>