import numpy as np
import random
from collections import deque
import time
from mesh_cutting import cut_mesh_growth_based, cut_mesh_shape_based


def cut_mesh(faces, centers, neighbors, min_hole_size, max_hole_size, method='growth'):

    if max_hole_size > len(faces):
        max_hole_size = len(faces)

    start_face = random.randint(0, len(faces) - 1)
    
    if method == 'growth':
        hole_size = random.randint(min_hole_size, max_hole_size)
        return cut_mesh_growth_based(neighbors, faces, start_face, hole_size)
    elif method == 'shape':
        hole_size = random.uniform(min_hole_size, max_hole_size)
        return cut_mesh_shape_based(faces, centers, neighbors, start_face, hole_size)


def write_stl(vertices, faces, normals, filename='output.stl'):
    with open(filename, 'w') as f:
        # Nagłówek pliku STL
        f.write("solid " + filename.split('.')[0] + "\n")

        # Zapisujemy trójkąty na podstawie współrzędnych wierzchołków
        for i in range(len(faces)):

            f.write("  facet normal {0} {1} {2}\n".format(normals[i][0], normals[i][1], normals[i][2]))
            f.write("    outer loop\n")

            for j in range(0, 3):
                f.write("      vertex {0} {1} {2}\n".format(vertices[faces[i][j]][0], vertices[faces[i][j]][1], vertices[faces[i][j]][2]))


            f.write("    endloop\n")
            f.write("  endfacet\n")

        # Zamknięcie pliku STL
        f.write("\nendsolid")

    print(f'Plik STL został wygenerowany: {filename}')



if __name__ == "__main__":
    FOLDER = "8d5aa0ca"
    START_VERTEX = 0
    HOLE_SIZE = 10000

    loading_start = time.time()
    Vertices = np.load(f"{FOLDER}/vertices.npy")
    Faces = np.load(f"{FOLDER}/faces.npy")
    Normals = np.load(f"{FOLDER}/normals.npy")
    Neighbors = np.load(f"{FOLDER}/neighbors.npy")
    Centers = np.load(f"{FOLDER}/centers.npy")
    loading_end = time.time()

    cutting_start = time.time()
    result_neighbors, result_faces, deleted = cut_mesh(Faces, Centers, Neighbors, 10, 50, method='shape')
    cutting_end = time.time()

    output_name = f"{FOLDER}_cut"
    output_fragemnt_name = f"{FOLDER}_fragment"

    generating_start = time.time()
    write_stl(Vertices, result_faces, Normals, output_name + ".stl")
    write_stl(Vertices, deleted, Normals, output_fragemnt_name + ".stl")
    generating_end = time.time()

    print(f"Loading time: {loading_end - loading_start} seconds")
    print(f"Cutting time: {cutting_end - cutting_start} seconds")
    print(f"Generating time: {generating_end - generating_start} seconds")
