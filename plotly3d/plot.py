import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def plot_3d(points, colors, s=1, alpha=1, force_continuous=False, title="3D Plot", filename=None):
    # Step 1: Determine Color Type
    unique_colors = np.unique(colors)
    is_categorical = (len(unique_colors) / len(colors) < 0.05) and not force_continuous # Heuristic threshold
    # Step 2: Preprocess Points
    scaler = MinMaxScaler()
    points_s = scaler.fit_transform(points)
    
    fig = go.Figure()
    
    if is_categorical:
        # Step 3: Categorical Colors Plotting Strategy
        for color in unique_colors:
            idx = colors == color
            fig.add_trace(go.Scatter3d(
                x=points_s[idx, 0],
                y=points_s[idx, 1],
                z=points_s[idx, 2],
                mode='markers',
                marker=dict(size=s, opacity=alpha, color=color),
                name=f'{color}'  # Customize for meaningful category names
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
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        showlegend=True
    )
    fig.show()
    if filename is not None:
        fig.write_html(filename)

    return fig
