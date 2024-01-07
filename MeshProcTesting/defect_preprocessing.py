from stl import mesh
import numpy as np
from itertools import combinations
import fast_simplification
import pandas as pd
import json
import time
from pathlib import Path
import requests
import os

MAX_VTCS = 50000

#function to get mesh geometric data from .stl file - vertices, faces, and so on
def get_mesh_data(filename, max_vertices):


    loaded_mesh = mesh.Mesh.from_file(filename)

    vertex_dict = {} #for storing vertices and their indices, helps to avoid duplicating the same vertices
    vertices = np.empty((0, 3), dtype=float) #array of vertices - shape (n, 3), stores x, y, z coordinates of each vertex
    faces = [] #array of faces - shape (n, 3), stores indices of vertices in each face
    centers = [] #array of centers - shape (n, 3), stores x, y, z coordinates of each face center
    corners = [] #array of corners - shape (n, 3, 3), stores x, y, z coordinates of each face corner group in faces

    normals = np.array(loaded_mesh.normals) #array of normals - shape (n, 3), stores x, y, z coordinates of each face's normal vector

    #calculating faces

    for vectors in loaded_mesh.vectors:
        for vertex in vectors:
            vertex_tuple = tuple(vertex)

            if vertex_tuple in vertex_dict:
                index = vertex_dict[vertex_tuple]
            else:
                index = len(vertices)
                vertices = np.append(vertices, [vertex], axis=0)
                vertex_dict[vertex_tuple] = index

        faces.append([vertex_dict[tuple(vertex)] for vertex in vectors])
        

    #calculating neighbors

    neighbors = [[] for _ in range(len(faces))]
    neighbors_dict = dict()

    for i in range(len(faces)):
        current_vertices = faces[i]
        splits = [frozenset([current_vertices[k], current_vertices[j]]) for k, j in combinations(range(len(current_vertices)), 2)]

        for current_split in splits:
            try:
                neighbors_dict[current_split].add(i)
            except KeyError:
                neighbors_dict[current_split] = set([i])

    for key, value in neighbors_dict.items():
        try:
            neighbor1, neighbor2 = value
            neighbors[neighbor1].append(neighbor2)
            neighbors[neighbor2].append(neighbor1)
        except ValueError:
            continue


    #ensuring the number of neighbors is 3 - array size check
    for i, neighbor_list in enumerate(neighbors):
        while len(neighbor_list) < 3:
            neighbor_list.append(i)

    #calculating centers and corners

    for i in range(len(faces)):
        face = np.array(faces[i])
        current_face_corners = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
        corners.append(current_face_corners)
        center = np.mean(current_face_corners, axis=0)
        centers.append(center.tolist())

    neighbors = np.array(neighbors)
    corners = np.array(corners)

    #saving to .npy files

    #np.save('vertices.npy', vertices)
    #np.save('faces.npy', faces)
    #np.save('neighbors.npy', neighbors)
    #np.save('centers.npy', centers)
    #np.save('normals.npy', normals)
    #np.save('corners.npy', corners)

    return vertices, faces, neighbors, centers, normals, corners

#a general preprocessing function

def prepare_data(csv_file):
    parent_folder = Path(__file__).resolve().parent

    dataset = pd.read_csv(csv_file)

    for subset in ['train', 'test', 'val']:
        dataset_folder = parent_folder / 'modified_dataset' / subset
        meshes_folder = dataset_folder / 'meshes'
        meshes_folder.mkdir(parents=True, exist_ok=True)
        meshlist = os.listdir(meshes_folder)
        fragments = []
        meshes = []
        for file_name in meshlist:
            if 'cut' in file_name:
                meshes.append(file_name)
            elif 'fragment' in file_name:
                fragments.append(file_name)
        meshdf = pd.DataFrame(meshes, columns=['mesh'])
        meshdf['id'] = meshdf['mesh'].str.split('.').str[0]
        meshdf['ground_truth'] = meshdf['mesh'].str.split('_').str[0]
        meshdf.to_csv(dataset_folder / 'summary.csv', index=False)
        summary_csv_file = dataset_folder / 'summary.csv'
        numpy_folder = dataset_folder / 'data'
        numpy_folder.mkdir(parents=True, exist_ok=True)
        meshdf.to_csv(summary_csv_file, index=False)

        fragmentsdf = pd.DataFrame(fragments, columns=['fragment'])
        fragmentsdf['id'] = fragmentsdf['fragment'].str.split('.').str[0]
        fragmentsdf['ground_truth'] = fragmentsdf['fragment'].str.split('_').str[0]
        fragments_folder = dataset_folder / 'fragments'
        fragments_folder.mkdir(parents=True, exist_ok=True)
        fragmentsdf.to_csv(fragments_folder / 'fragments.csv', index=False)
        fragments_numpy_folder = fragments_folder / 'data'
        fragments_numpy_folder.mkdir(parents=True, exist_ok=True)

        print(f"Preparing {subset} dataset...")

        for case in ['meshes', 'fragments']:

            if case == 'meshes':
                for index, row in meshdf.iterrows():
                    start = time.time()
                    print("id: ", row['id'])
                    mesh_file = meshes_folder / row['mesh']
                    json_filename = str(row['id']) + '.json'
                    json_file = dataset_folder / json_filename

                    #try:
                    print(mesh_file)
                    vertices, faces, neighbors, centers, normals, corners = get_mesh_data(mesh_file, MAX_VTCS)
                    mesh_folder = numpy_folder / str(row['id'])
                    mesh_folder.mkdir(parents=True, exist_ok=True)
                    np.save(mesh_folder / 'vertices.npy', vertices)
                    np.save(mesh_folder / 'faces.npy', faces)
                    np.save(mesh_folder / 'corners.npy', corners)
                    np.save(mesh_folder / 'centers.npy', centers)
                    np.save(mesh_folder / 'normals.npy', normals)
                    np.save(mesh_folder / 'neighbors.npy', neighbors)

                    # Zapisz plik JSON tylko jeśli powiodło się pobieranie i przetwarzanie danych
                    json_content = {
                            "name": str(row['id']),
                            "vertices": str(mesh_folder / 'vertices.npy'),
                            "faces": str(mesh_folder / 'faces.npy'),
                            "corners": str(mesh_folder / 'corners.npy'),
                            "centers": str(mesh_folder / 'centers.npy'),
                            "normals": str(mesh_folder / 'normals.npy'),
                            "neighbors": str(mesh_folder / 'neighbors.npy'),
                            "mesh": str(mesh_file)
                    }
                    json_content = json.dumps(json_content)
                    with json_file.open('w') as f:
                        f.write(json_content)
                    print("Zapisano plik json")
                    #except (IndexError, AttributeError, ValueError, AssertionError, ZeroDivisionError) as e:
                        #print(f"Błąd w przetwarzaniu danych: {e}. Pominięto dane o id: {row['id']}")
                    stop = time.time()
                    print("Czas operacji: ", stop - start)
            
            elif case == 'fragments':
                for index, row in meshdf.iterrows():
                    start = time.time()
                    print("id: ", row['id'])
                    mesh_file = meshes_folder / row['mesh']
                    json_filename = str(row['id']) + '.json'
                    json_file = fragments_folder / json_filename

                    #try:
                    print(mesh_file)
                    vertices, faces, neighbors, centers, normals, corners = get_mesh_data(mesh_file, MAX_VTCS)
                    mesh_folder = fragments_numpy_folder / str(row['id'])
                    mesh_folder.mkdir(parents=True, exist_ok=True)
                    np.save(mesh_folder / 'vertices.npy', vertices)
                    np.save(mesh_folder / 'faces.npy', faces)
                    np.save(mesh_folder / 'corners.npy', corners)
                    np.save(mesh_folder / 'centers.npy', centers)
                    np.save(mesh_folder / 'normals.npy', normals)
                    np.save(mesh_folder / 'neighbors.npy', neighbors)

                    # Zapisz plik JSON tylko jeśli powiodło się pobieranie i przetwarzanie danych
                    json_content = {
                            "name": str(row['id']),
                            "vertices": str(mesh_folder / 'vertices.npy'),
                            "faces": str(mesh_folder / 'faces.npy'),
                            "corners": str(mesh_folder / 'corners.npy'),
                            "centers": str(mesh_folder / 'centers.npy'),
                            "normals": str(mesh_folder / 'normals.npy'),
                            "neighbors": str(mesh_folder / 'neighbors.npy'),
                            "mesh": str(mesh_file)
                    }
                    json_content = json.dumps(json_content)
                    with json_file.open('w') as f:
                        f.write(json_content)
                    print("Zapisano plik json")
                    #except (IndexError, AttributeError, ValueError, AssertionError, ZeroDivisionError) as e:
                        #print(f"Błąd w przetwarzaniu danych: {e}. Pominięto dane o id: {row['id']}")
                    stop = time.time()
                    print("Czas operacji: ", stop - start)
                

        


if __name__ == "__main__":

    CSV = 'dataset.csv'
    prepare_data(CSV)