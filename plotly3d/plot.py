import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os

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
    is_3d = points.shape[1] == 3
    plot_func = go.Scatter3d if is_3d else go.Scatter
    scaler = kwargs.get('scaler', None)
    s = kwargs.get('s', 1)
    alpha = kwargs.get('alpha', 1)
    title = kwargs.get('title', 'Plot')
    filename = kwargs.get('filename', None)
    xtitle = kwargs.get('xtitle', 'X')
    ytitle = kwargs.get('ytitle', 'Y')
    ztitle = kwargs.get('ztitle', 'Z')
    force_continuous = kwargs.get('force_continuous', False)
    rescale = kwargs.get('rescale', True)
    legend = kwargs.get('legend', True)
    colorscale = kwargs.get('colorscale', 'Viridis')
    fig = kwargs.get('fig', go.Figure())
    points = np.asarray(points)
    colors = np.asarray(colors) if colors is not None else None
    if rescale:
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(points)
        points_s = scaler.transform(points)
    else:
        points_s = points

    if colors is None:
        colors = np.zeros(points.shape[0])
        is_categorical = True
    else:
        # Step 1: Determine Color Type
        unique_colors = np.unique(colors)
        is_categorical = (len(unique_colors) / len(colors) < 0.05) and not force_continuous # Heuristic threshold
        # Step 2: Preprocess Points
    if is_categorical:
        # Step 3: Categorical Colors Plotting Strategy
        # Map categorical color labels to integers
        color_map, categories = pd.factorize(colors, sort=True)
        # Create a trace for each unique color/category
        for i, color in enumerate(categories):
            idx = color_map == i
            if is_3d:
                fig.add_trace(plot_func(
                    x=points_s[idx, 0],
                    y=points_s[idx, 1],
                    z=points_s[idx, 2],
                    mode='markers',
                    marker=dict(size=s, opacity=alpha, color=i), # Use integer mapping for color
                    name=str(color)  # Use actual category name for legend
                ))
            else:
                fig.add_trace(plot_func(
                    x=points_s[idx, 0],
                    y=points_s[idx, 1],
                    mode='markers',
                    marker=dict(size=s, opacity=alpha, color=i), # Use integer mapping for color
                    name=str(color)  # Use actual category name for legend
                ))
    else:
        # Step 3: Continuous Colors Plotting Strategy
        if is_3d:
            fig.add_trace(plot_func(
                x=points_s[:, 0],
                y=points_s[:, 1],
                z=points_s[:, 2],
                mode='markers',
                marker=dict(size=s, color=colors, colorscale='Viridis', opacity=alpha, colorbar=dict(title='Color Scale')),
            ))
        else:
            fig.add_trace(plot_func(
                x=points_s[:, 0],
                y=points_s[:, 1],
                mode='markers',
                marker=dict(size=s, color=colors, colorscale='Viridis', opacity=alpha, colorbar=dict(title='Color Scale')),
            ))
    
    fig.data[0].marker.colorscale = colorscale
    if is_3d:
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title=xtitle, yaxis_title=ytitle, zaxis_title=ztitle),
            showlegend=legend
        )
    else:
        fig.update_layout(
            title=title,
            xaxis_title=xtitle, yaxis_title=ytitle,
            showlegend=legend
        )
    if filename is not None:
        fig.write_html(filename)

    return fig

# for compatability with previous versions
plot_3d = scatter
scatter3d = scatter

# def trajectories(trajs, s=1, alpha=1, title="3D Plot", filename=None, rescale=True, fig=None, xtitle='X', ytitle='Y', ztitle='Z', scaler=None, legend_label="Trajectories"):
def trajectories(trajs, **kwargs):
    """
    Plots trajectories in 3D space using Plotly, with thinner gray lines without markers and reduced line opacity.

    Parameters:
    - trajs: numpy array of shape (a, b, c), where
      a: time bins, b: trajectory index, c: space dimension (c=3 for 3D).

    Returns:
    - Plotly figure containing the trajectories plotted in 3D space.
    """
    is_3d = trajs.shape[2] == 3
    plot_func = go.Scatter3d if is_3d else go.Scatter
    s = kwargs.get('s', 1)
    s_end = kwargs.get('s_end', 1)
    alpha = kwargs.get('alpha', 1)
    title = kwargs.get('title', 'Plot')
    filename = kwargs.get('filename', None)
    rescale = kwargs.get('rescale', True)
    fig = kwargs.get('fig', go.Figure())
    xtitle = kwargs.get('xtitle', 'X')
    ytitle = kwargs.get('ytitle', 'Y')
    ztitle = kwargs.get('ztitle', 'Z')
    scaler = kwargs.get('scaler', None)
    legend_label = kwargs.get('legend_label', 'Trajectories')

    trajs = np.asarray(trajs)
    if rescale:
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(trajs.reshape(-1, trajs.shape[2]))
        trajs_s = scaler.transform(trajs.reshape(-1, trajs.shape[2])).reshape(trajs.shape)
    else:
        trajs_s = trajs

    # Number of trajectories
    num_trajectories = trajs.shape[1]
    
    # Use a legendgroup and make the dummy trace for the legend entry
    legendgroup = "trajectories"
    # Add a dummy trace for the legend entry, enabling its visibility toggles the group
    if is_3d:
        fig.add_trace(plot_func(
            x=[None],
            y=[None],
            z=[None],
            mode='lines',
            line=dict(color='gray', width=s),
            name=legend_label,
            legendgroup=legendgroup,
            # This is the key for toggling visibility
            visible=True
        ))
    else:
        fig.add_trace(plot_func(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color='gray', width=s),
            name=legend_label,
            legendgroup=legendgroup,
            # This is the key for toggling visibility
            visible=True
        ))
    # Iterate over each trajectory and plot it, linked by legendgroup
    for i in range(num_trajectories):
        trajectory = trajs_s[:, i, :]
        if is_3d:
            fig.add_trace(plot_func(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode='lines',
                line=dict(width=s, color='gray'),
                opacity=alpha,
                showlegend=False,  # Do not show these individual traces in the legend
                legendgroup=legendgroup,  # Link visibility control to the dummy trace
                visible=True  # Initially visible
            ))
        else:
            fig.add_trace(plot_func(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                mode='lines',
                line=dict(width=s, color='gray'),
                opacity=alpha,
                showlegend=False,  # Do not show these individual traces in the legend
                legendgroup=legendgroup,  # Link visibility control to the dummy trace
                visible=True  # Initially visible
            ))
        # Add endpoint marker
        if is_3d:
            fig.add_trace(plot_func(
                x=[trajectory[-1, 0]],
                y=[trajectory[-1, 1]],
                z=[trajectory[-1, 2]],
                mode='markers',
                marker=dict(size=s_end, color='red'), # Customize color and size as needed
                showlegend=False,
                legendgroup=legendgroup,
                visible=True
            ))
        else:
            fig.add_trace(plot_func(
                x=[trajectory[-1, 0]],
                y=[trajectory[-1, 1]],
                mode='markers',
                marker=dict(size=s_end, color='red'), # Customize color and size as needed
                showlegend=False,
                legendgroup=legendgroup,
                visible=True
            ))

    # Set the layout for the 3D plot
    if is_3d:
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=xtitle, yaxis_title=ytitle, zaxis_title=ztitle,
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
    else:
        fig.update_layout(
            title=title,
            xaxis_title=xtitle, yaxis_title=ytitle,
            margin=dict(l=0, r=0, b=0, t=30)
        )
    if filename is not None:
        fig.write_html(filename)

    return fig