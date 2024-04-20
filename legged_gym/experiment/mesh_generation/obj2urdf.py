import trimesh
import xml.etree.ElementTree as ET
import base64
import numpy as np
import json


names_and_positions = json.load(open('parkour_meshes/rectangular_prisms.json'))

def obj_to_urdf(obj_file_path, urdf_file_path, xy_pos, name='mesh'):
    # Load OBJ file
    mesh = trimesh.load_mesh(obj_file_path)

    # Create URDF XML structure
    urdf_root = ET.Element("robot")
    urdf_root.set("name", name)

    # Create link element
    link = ET.SubElement(urdf_root, "link")
    link.set("name", name)

    # Create visual element
    visual = ET.SubElement(link, "visual")
    visual_origin = ET.SubElement(visual, "origin")
    visual_origin.set("xyz", f"{xy_pos[0]} {xy_pos[1]} 0")
    visual_origin.set("rpy", "0 0 0")
    visual_geometry = ET.SubElement(visual, "geometry")

    # Check if mesh is a trimesh.Trimesh object
    if isinstance(mesh, trimesh.Trimesh):
        # Extract vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces

        # Encode vertices and faces as base64 strings
        vertices_str = base64.b64encode(vertices.tobytes()).decode()
        faces_str = base64.b64encode(faces.tobytes()).decode()

        # Create mesh element
        mesh_element = ET.SubElement(visual_geometry, "mesh")
        mesh_element.set("filename", name + ".obj")

        # Create material element
        material = ET.SubElement(visual, "material")
        material.set("name", "Material")
        color = ET.SubElement(material, "color")
        color.set("rgba", "0.5 0.5 0.5 1")

    # Generate URDF XML file
    urdf_tree = ET.ElementTree(urdf_root)
    with open(urdf_file_path, "wb") as f:
        urdf_tree.write(f)


for i in names_and_positions['rectangular_prisms']:
    filepath = i['filename']
    pos = i['position']

    urdf_filepath = filepath.replace('sep', 'urdf').replace('.obj','.urdf')
    obj_to_urdf(filepath, urdf_filepath,pos)


filepath = '/home/andrewjenkins/extreme-parkour/legged_gym/experiment/mesh_generation/parkour_meshes/rectangular_prism_bumps.obj'
pos = [0,0]
urdf_filepath = filepath.replace('sep', 'urdf').replace('.obj','.urdf')
obj_to_urdf(filepath, urdf_filepath,pos)