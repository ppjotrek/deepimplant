import numpy as np

def generate_sphere(radius, num_faces):
    vertices = []
    faces = []

    for i in range(num_faces):
        theta = 2 * np.pi * i / num_faces
        phi = np.pi * (i + 0.5) / num_faces

        x = radius * np.cos(theta) * np.sin(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(phi)

        vertices.append([x, y, z])

    # Add the top and bottom vertices
    vertices.append([0, 0, radius])
    vertices.append([0, 0, -radius])

    # Generate faces for the sides
    for i in range(num_faces):
        faces.append([i, (i + 1) % num_faces, num_faces])
        faces.append([i, num_faces + 1, (i + 1) % num_faces])

    return vertices, faces

def write_stl_file(vertices, faces, filename):
    with open(filename, 'w') as f:
        f.write("solid Sphere\n")
        for face in faces:
            normal = np.cross(
                np.array(vertices[face[1]]) - np.array(vertices[face[0]]),
                np.array(vertices[face[2]]) - np.array(vertices[face[0]])
            )
            normal /= np.linalg.norm(normal)
            f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
            f.write("    outer loop\n")
            for vertex_index in face:
                vertex = vertices[vertex_index]
                f.write(f"      vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid Sphere\n")

# Ustawienia sfery
radius = 1.0
num_faces = 20
output_filename = "sphere.stl"

# Generuj sfery
vertices, faces = generate_sphere(radius, num_faces)

# Zapisz do pliku STL
write_stl_file(vertices, faces, output_filename)

print(f"Plik STL {output_filename} został wygenerowany.")
