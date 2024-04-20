import json
import trimesh
import plotly.graph_objects as go
import numpy as np
import random

# Load the rectangular prisms data from json
with open('parkour_meshes/rectangular_prisms.json', 'r') as infile:
    data = json.load(infile)

# Create a figure
fig = go.Figure()

# Load and add each rectangular prism to the figure
for prism_data in data['rectangular_prisms']:
    filename = prism_data['filename']
    position = prism_data['position']

    # Load the obj file
    mesh = trimesh.load(filename)
    
     # Generate a random color for the mesh
    color = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"

    # Create a trace for the mesh
    trace = go.Mesh3d(
        x=mesh.vertices[:, 0] + position[0],
        y=mesh.vertices[:, 1] + position[1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color=color,
        opacity=1.0,
        flatshading=True,
    )

    # Add the trace to the figure
    fig.add_trace(trace)

# Set the layout for the figure
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data',
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(
            eye=dict(x=0, y=0, z=10),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=1, z=0)
        ),
    ),
    width=800,
    height=600,
    margin=dict(l=0, r=0, t=0, b=0)
)

# Display the figure
fig.write_image('./rendered_mesh.jpg')