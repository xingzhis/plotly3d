import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def scatter3d(points, colors=None, scaler=None, **kwargs):
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
    points = np.asarray(points)
    colors = np.asarray(colors) if colors is not None else None
    rescale = kwargs.get('rescale', True)
    if rescale:
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(points)
        points_s = scaler.transform(points)
    else:
        points_s = points

    fig = kwargs.get('fig', go.Figure())

    s = kwargs.get('s', 1)
    alpha = kwargs.get('alpha', 1)
    title = kwargs.get('title', '3D Plot')
    filename = kwargs.get('filename')
    xtitle = kwargs.get('xtitle', 'X')
    ytitle = kwargs.get('ytitle', 'Y')
    ztitle = kwargs.get('ztitle', 'Z')
    force_continuous = kwargs.get('force_continuous', False)

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
            fig.add_trace(go.Scatter3d(
                x=points_s[idx, 0],
                y=points_s[idx, 1],
                z=points_s[idx, 2],
                mode='markers',
                marker=dict(size=s, opacity=alpha, color=i), # Use integer mapping for color
                name=str(color)  # Use actual category name for legend
            ))
    else:
        # Step 3: Continuous Colors Plotting Strategy
        fig.add_trace(go.Scatter3d(
            x=points_s[:, 0],
            y=points_s[:, 1],
            z=points_s[:, 2],
            mode='markers',
            marker=dict(size=s, color=colors, colorscale='Viridis', opacity=alpha, colorbar=dict(title='Color Scale')),
        ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=xtitle, yaxis_title=ytitle, zaxis_title=ztitle),
        showlegend=True
    )
    if filename is not None:
        fig.write_html(filename)

    return fig

# for compatability with previous versions
plot_3d = scatter3d

def trajectories(trajs, s=1, alpha=1, title="3D Plot", filename=None, rescale=True, fig=None, xtitle='X', ytitle='Y', ztitle='Z', scaler=None):
    """
    Plots trajectories in 3D space using Plotly, with thinner gray lines without markers and reduced line opacity.

    Parameters:
    - trajs: numpy array of shape (a, b, c), where
      a: time bins, b: trajectory index, c: space dimension (c=3 for 3D).

    Returns:
    - Plotly figure containing the trajectories plotted in 3D space.
    """
    trajs = np.asarray(trajs)
    # Create a Plotly figure
    if fig is None:
        fig = go.Figure()
    
    if rescale:
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(trajs.reshape(-1, 3))
        trajs_s = scaler.transform(trajs.reshape(-1, 3)).reshape(trajs.shape)
    else:
        trajs_s = trajs
    # Number of trajectories
    num_trajectories = trajs.shape[1]
    
    # Iterate over each trajectory and plot it
    for i in range(num_trajectories):
        # Extract the trajectory trajs
        trajectory = trajs_s[:, i, :]
        
        # Add the trajectory to the plot with specified line properties
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0],  # X coordinates
            y=trajectory[:, 1],  # Y coordinates
            z=trajectory[:, 2],  # Z coordinates
            mode='lines',  # Only use lines (no markers)
            line=dict(
                width=s,  # Make the line thinner
                color='gray',  # Set the line color to gray
            ),
            opacity=alpha,  # Reduce the opacity of the lines
            showlegend=False
        ))
    
    # Set the layout for the 3D plot
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=xtitle, yaxis_title=ytitle, zaxis_title=ztitle
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    if filename is not None:
        fig.write_html(filename)

    return fig