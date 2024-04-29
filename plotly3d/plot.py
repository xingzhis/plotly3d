import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
import plotly.express as px
import matplotlib.cm as cm


def scatter(points, colors=None, **kwargs):
    '''
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
    '''

    is_3d = points.shape[1] == 3
    plot_func = go.Scatter3d if is_3d else lambda z, **kwargs: go.Scatter(**kwargs)
    scaler = kwargs.get('scaler', None)
    s = kwargs.get('s', 1)
    alpha = kwargs.get('alpha', 1)
    title = kwargs.get('title', 'Plot')
    filename = kwargs.get('filename', None)
    xtitle = kwargs.get('xtitle', 'X')
    ytitle = kwargs.get('ytitle', 'Y')
    ztitle = kwargs.get('ztitle', 'Z')
    force_continuous = kwargs.get('force_continuous', False)
    rescale = kwargs.get('rescale', False)
    legend = kwargs.get('legend', True)
    colorscale = kwargs.get('colorscale', 'Viridis')
    fig = kwargs.get('fig', go.Figure())
    points = np.asarray(points)
    colors = np.asarray(colors) if colors is not None else None
    ticks = kwargs.get('ticks', True)
    figsize = kwargs.get('figsize', None)
    white_bkgrnd = kwargs.get('white_bkgrnd', False)

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
            fig.add_trace(plot_func(
                x=points_s[idx, 0],
                y=points_s[idx, 1],
                z=points_s[idx, 2] if is_3d else None,
                mode='markers',
                marker=dict(size=s, opacity=alpha, color=i), # Use integer mapping for color
                name=str(color)  # Use actual category name for legend
            ))

    else:
        # Step 3: Continuous Colors Plotting Strategy
        fig.add_trace(plot_func(
            x=points_s[:, 0],
            y=points_s[:, 1],
            z=points_s[:, 2] if is_3d else None,
            mode='markers',
            marker=dict(size=s, color=colors, colorscale='Viridis', opacity=alpha, colorbar=dict(title='Color Scale')),
        ))

    fig.data[0].marker.colorscale = colorscale
    scene=dict(xaxis_title=xtitle, yaxis_title=ytitle)
    if is_3d:
        scene['zaxis_title'] = ztitle
    fig.update_layout(
        title=title,
        scene=scene,
        showlegend=legend
    )

    if not ticks:
        fig.update_xaxes(showgrid=False, zeroline=False, showline=False, showticklabels=False, ticks="")
        fig.update_yaxes(showgrid=False, zeroline=False, showline=False, showticklabels=False, ticks="")
        if is_3d:
            fig.update_zaxes(showgrid=False, zeroline=False, showline=False, showticklabels=False, ticks="")

    if figsize is not None:
        assert len(figsize) == 2
        fig.update_layout(width=figsize[0] * 100, height=figsize[1] * 100)  # width and height in pixels

    if white_bkgrnd:
        fig.update_layout(
            paper_bgcolor='white',  # Color of the whole background
            plot_bgcolor='white'    # Color of the plotting area
        )

    if filename is not None:
        fig.write_html(filename)

    return fig

# for compatability with previous versions
plot_3d = scatter
scatter3d = scatter

def trajectories(trajs, colors=None, **kwargs):
    '''
    Plots trajectories in 3D space using Plotly, with unique colors for each category and a single legend entry per category.

    Parameters:
    - trajs: numpy array of shape (a, b, c), where
      a: time bins, b: trajectory index, c: space dimension (c=3 for 3D).

    Returns:
    - Plotly figure containing the trajectories plotted in 3D space.
    '''

    is_3d = trajs.shape[2] == 3
    plot_func = go.Scatter3d if is_3d else lambda z, **kwargs: go.Scatter(**kwargs)
    s = kwargs.get('s', 1)
    s_end = kwargs.get('s_end', 1)
    title = kwargs.get('title', 'Plot')
    filename = kwargs.get('filename', None)
    rescale = kwargs.get('rescale', False)
    fig = kwargs.get('fig', go.Figure())
    xtitle = kwargs.get('xtitle', 'X')
    ytitle = kwargs.get('ytitle', 'Y')
    ztitle = kwargs.get('ztitle', 'Z')
    scaler = kwargs.get('scaler', None)
    # colors = kwargs.get('colors', None)
    cmap = kwargs.get('cmap', 'tab20')
    ticks = kwargs.get('ticks', True)
    figsize = kwargs.get('figsize', None)
    white_bkgrnd = kwargs.get('white_bkgrnd', False)

    if colors is None:
        colors = np.zeros(trajs.shape[1])
    # color_palette = kwargs.get('color_palette', px.colors.qualitative.Plotly)  # Define a color palette

    trajs = np.asarray(trajs)
    if rescale and scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(trajs.reshape(-1, trajs.shape[2]))
    trajs_s = scaler.transform(trajs.reshape(-1, trajs.shape[2])).reshape(trajs.shape) if rescale else trajs

    color_map, categories = pd.factorize(colors, sort=True)
    cmap = cm.get_cmap(cmap, len(categories))  # Get a colormap with as many colors as categories

    # color_map = color_map % len(color_palette)  # Ensure we use the color palette cyclically if not enough colors

    for i, color in enumerate(categories):
        category_color = cmap(i % 20)[:3]  # Get color from colormap, repeat if more than 20 categories
        category_color = 'rgb' + str(tuple(int(c*255) for c in category_color))  # Convert to RGB string for Plotly
        idx = color_map == i
        first_idx = np.where(idx)[0][0]
        for j in np.where(idx)[0]:
            trajectory = trajs_s[:, j, :]
            showlegend = (j == first_idx)  # Compare indices to determine if it's the first trajectory
            showlegend = bool(showlegend)  # Explicitly convert to Python bool

            # Add the trace for the trajectory
            fig.add_trace(plot_func(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2] if is_3d else None,
                mode= 'lines',  # Add markers only for the first trajectory
                line=dict(width=s, color=category_color),
                name=str(color) if showlegend else '',  # Only set the name for the first trajectory
                showlegend=showlegend,
                legendgroup=str(color)
            ))

            fig.add_trace(plot_func(
                x=[trajectory[-1, 0]],
                y=[trajectory[-1, 1]],
                z=[trajectory[-1, 2]] if is_3d else None,
                mode='markers',
                marker=dict(size=s_end, color='black'),  # Black dot
                showlegend=False,
                legendgroup=str(color)
            ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=xtitle, yaxis_title=ytitle, zaxis_title=ztitle) if is_3d else dict(),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    if not ticks:
        fig.update_xaxes(showgrid=False, zeroline=False, showline=False, showticklabels=False, ticks="")
        fig.update_yaxes(showgrid=False, zeroline=False, showline=False, showticklabels=False, ticks="")
        if is_3d:
            fig.update_zaxes(showgrid=False, zeroline=False, showline=False, showticklabels=False, ticks="")

    if figsize is not None:
        assert len(figsize) == 2
        fig.update_layout(width=figsize[0] * 100, height=figsize[1] * 100)  # width and height in pixels

    if white_bkgrnd:
        fig.update_layout(
            paper_bgcolor='white',  # Color of the whole background
            plot_bgcolor='white'        # Color of the plotting area
        )

    if filename is not None:
        fig.write_html(filename)

    return fig

