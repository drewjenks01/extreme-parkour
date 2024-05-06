def generate_texture_coordinates(vertices, scale_factor):
    # Calculate bounding box of the vertices
    min_x = min(vertex[0] for vertex in vertices)
    max_x = max(vertex[0] for vertex in vertices)
    min_y = min(vertex[1] for vertex in vertices)
    max_y = max(vertex[1] for vertex in vertices)
    min_z = min(vertex[2] for vertex in vertices)
    max_z = max(vertex[2] for vertex in vertices)

    # Map vertices to texture coordinates
    texture_vertices = []
    for vertex in vertices:
        u = scale_factor * (vertex[0] - min_x) / (max_x - min_x)
        v = scale_factor * (vertex[1] - min_y) / (max_y - min_y)
        w = scale_factor * (vertex[2] - min_z) / (max_z - min_z)

        texture_vertices.append((u, v, w))
    
    return texture_vertices

def generate_rectangular_prism(vertices, triangles, filepath):
    texture_vertices = generate_texture_coordinates(vertices, 50)
    with open(filepath, 'w') as f:
        # Write vertices
        # f.write('# Blender 3.1.2\n')
        # f.write('# www.blender.org\n')
        # f.write('mtllib rectangular_prism_bumps_whole.mtl\n')
        for vertex in vertices:
            f.write("v {} {} {}\n".format(vertex[0], vertex[1], vertex[2]))
        
        # Write texture vertices
        for texture_vertex in texture_vertices:
            f.write("vt {} {}\n".format(texture_vertex[0], texture_vertex[1]))
        
        # Write triangles with texture indices
        for triangle in triangles:
            # Assume that texture vertices are assigned in the same order as vertices
            f.write("f {}/{} {}/{} {}/{}\n".format(
                triangle[0] + 1, triangle[0] + 1,
                triangle[1] + 1, triangle[1] + 1,
                triangle[2] + 1, triangle[2] + 1
            ))
