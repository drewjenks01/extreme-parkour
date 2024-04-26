import math
import numpy as np

def generate_rectangular_prism(L, W, H, nl, nw, nh, terrain_heights, filename):
    vertices = []
    faces = []

    # Generate vertices for each surface
    for i in range(nl + 1):
        for j in range(nw + 1):
            x = L * i / nl - L / 2
            y = W * j / nw - W / 2
            vertices.append((x, y, -H))
            
            # Add bumps on the top surface
            vertices.append((x, y, terrain_heights[i, j]))

    for i in range(nl + 1):
        for k in range(nh + 1):
            x = L * i / nl - L / 2
            z = H * k / nh - H
            vertices.append((x, -W / 2, z))
            vertices.append((x, W / 2, z))

    for j in range(nw + 1):
        for k in range(nh + 1):
            y = W * j / nw - W / 2
            z = H * k / nh - H
            vertices.append((-L / 2, y, z))
            vertices.append((L / 2, y, z))

    # Generate faces for each surface
    # Bottom and top faces
    for i in range(nl):
        for j in range(nw):
            idx1 = (i * (nw + 1) + j) * 2
            idx2 = idx1 + 2
            idx3 = idx1 + (nw + 1) * 2
            idx4 = idx3 + 2
            faces.append((idx1, idx2, idx4, idx3))
            faces.append((idx1 + 1, idx3 + 1, idx4 + 1, idx2 + 1))

    offset = (nl + 1) * (nw + 1) * 2
    # Front and back faces
    for i in range(nl):
        for k in range(nh):
            idx1 = offset + (i * (nh + 1) + k) * 2
            idx2 = idx1 + 2
            idx3 = idx1 + (nh + 1) * 2
            idx4 = idx3 + 2
            faces.append((idx1, idx2, idx4, idx3))
            faces.append((idx1 + 1, idx3 + 1, idx4 + 1, idx2 + 1))

    offset += (nl + 1) * (nh + 1) * 2
    # Left and right faces
    for j in range(nw):
        for k in range(nh):
            idx1 = offset + (j * (nh + 1) + k) * 2
            idx2 = idx1 + 2
            idx3 = idx1 + (nh + 1) * 2
            idx4 = idx3 + 2
            faces.append((idx1, idx2, idx4, idx3))
            faces.append((idx1 + 1, idx3 + 1, idx4 + 1, idx2 + 1))

    with open(filename, 'w') as f:
        f.write('# Blender 3.1.2\n')
        f.write('# www.blender.org\n')
        f.write('mtllib rectangular_prism_bumps.mtl\n')
        f.write(f'o {filename.split("/")[-1].replace(".obj", "")}\n')
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            f.write(f"f {face[0]+1} {face[2]+1} {face[3]+1}\n")

# # Usage example
# # L = 2.0
# # W = 1.5
# H = 1.5
# # nl = 10
# # nw = 8
# nh = 6
# filename = "parkour_meshes/rectangular_prism_bumps.obj"

# import os
# os.makedirs("parkour_meshes", exist_ok=True)
# os.makedirs("parkour_meshes/sep", exist_ok=True)

# from parkour_terrain import Cfg, Terrain

# cfg = Cfg()
# terrain_obj = Terrain(cfg, num_robots=2000)

# terrain_heights = terrain_obj.height_field_raw * cfg.vertical_scale
# nl, nw = terrain_heights.shape[0] - 1, terrain_heights.shape[1] - 1
# W, L = cfg.num_rows * cfg.terrain_width, cfg.num_cols * cfg.terrain_length

# # generate the full map
# # seal the edges
# terrain_heights[:, 0] = 0
# terrain_heights[:, -1] = 0
# terrain_heights[0, :] = 0
# terrain_heights[-1, :] = 0

# generate_rectangular_prism(L, W, H, nl, nw, nh, terrain_heights, filename)

# # for each terrain section, generate a rectangular prism
# for row in range(cfg.num_rows):
#     for col in range(cfg.num_cols):
#         filename = f"parkour_meshes/sep/rectangular_prism_bumps_{row}_{col}.obj"
#         terrain_heights = terrain_obj.height_field_raw[row*terrain_obj.length_per_env_pixels:(row+1)*terrain_obj.length_per_env_pixels, col*terrain_obj.width_per_env_pixels:(col+1)*terrain_obj.width_per_env_pixels] * cfg.vertical_scale
#         terrain_heights[:, 0] = 0
#         terrain_heights[:, -1] = 0
#         terrain_heights[0, :] = 0
#         terrain_heights[-1, :] = 0
#         nl, nw = terrain_heights.shape[0] - 1, terrain_heights.shape[1] - 1
#         L, W = cfg.terrain_length, cfg.terrain_width
#         generate_rectangular_prism(L, W, H, nl, nw, nh, terrain_heights, filename)
        
# # write a json file with the locations of the rectangular prisms
# import json
# data = {}
# data['rectangular_prisms'] = []
# for row in range(cfg.num_rows):
#     for col in range(cfg.num_cols):
#         filename = f"parkour_meshes/sep/rectangular_prism_bumps_{row}_{col}.obj"
#         data['rectangular_prisms'].append({
#             'filename': filename,
#             'position': [row * cfg.terrain_length, col * cfg.terrain_width]
#         })

# with open('parkour_meshes/rectangular_prisms.json', 'w') as outfile:
#     json.dump(data, outfile)
        
