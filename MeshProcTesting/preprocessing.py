from pathlib import Path
import pandas as pd
import requests
import numpy as np
import json
import shutil
import time
import scipy.spatial
from scipy.spatial import Delaunay
import fast_simplification
from stl import mesh

CONFIG = {
    'max_vertices' : 50000
}

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Pobrano i zapisano plik jako {save_path}")
    else:
        print(f"Wystąpił błąd podczas pobierania pliku {save_path}")

def read_obj_file(path: str):
    vertices = []
    faces = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith('v'):
                line = line[1:].split()
                vertices.append([float(line[i]) for i in range(3)])
            elif line.startswith('f'):
                line = line[1:].split()
                faces.append([int(line[i]) for i in range(4)])
    
    return np.array(vertices), np.array(faces)

def read_stl_file(path: str, return_normals=False):
    vertices = []
    faces = []
    normals = []

    with open(path, "r") as f:
        for line in f:
            if line.strip().startswith('vertex'):
                vertex = [float(coord) for coord in line.strip().split()[1:]]
                vertices.append(vertex)
            elif line.strip().startswith('endloop'):
                face = [len(vertices) - 3, len(vertices) - 2, len(vertices) - 1]
                faces.append(face)
            elif line.strip().startswith('facet normal'):
                normal = [float(coord) for coord in line.strip().split()[2:]]
                normals.append(normal)
    
    vertices = np.array(vertices)
    faces = np.array(faces)

    if return_normals:
        return vertices, faces, normals
    else:
        return vertices, faces

class Mesh:
    def __init__(self, path: str, max_vertices=None): 
        if path.suffix == '.obj':
            your_mesh = mesh.Mesh.from_file(path)
            vertices = your_mesh.points
            self.vertices, self.faces = read_obj_file(path)
            self.fileformat = 'obj'
        elif path.suffix == '.stl':
            your_mesh = mesh.Mesh.from_file(path)
            self.faces = your_mesh.vectors
            self.normals = your_mesh.normals
            self.vertices = your_mesh.points
            #self.vertices, self.faces, self.normals = read_stl_file(path, return_normals=True)
            self.fileformat = 'stl'
        else:
            raise ValueError('Unsupported file format. Supported formats are .obj and .stl. Format you tried to use is: ', path.suffix)
        '''
        if max_vertices is not None:
            self.max_vertices = max_vertices
            if len(self.vertices) > self.max_vertices:
                self.vertices = self.vertices.astype(np.float32)

                reduction = 1.0 - (self.max_vertices / len(self.vertices))

                new_vtcs, new_faces = fast_simplification.simplify(self.vertices, self.faces, reduction, return_collapses = False)

                self.vtcs = new_vtcs
                self.faces = new_faces
'''
#downsampling nie jest problemem, więc zostawić, ale przyjrzeć się czy nie psuje
        self.path = path
        self.triangle_mesh = self.triangulate_faces() # do wyjebania bo nie potrzebne przy .stl
    
    def triangulate_faces(self):
        if self.fileformat == 'stl':
            return self.faces
        else:
            triangles = []
            for face in self.faces:
                if len(face) > 3:
                    for i in range(1, len(face) - 1):
                        triangle = [face[0]-1, face[i]-1, face[i + 1]-1]
                        triangles.append(triangle)
                else:
                    triangles.append(face)
            self.triangles = triangles
            return triangles
    
    def get_triangulated_faces(self):
        return self.triangle_mesh

    def get_vertices(self):
        return self.vertices
    
    def get_faces(self):
        return self.faces
    
    def get_path(self):
        return self.path
    
    def get_corners(self):
        corners = []

        for face in self.triangle_mesh:
            if len(face) == 3:
                triangle = [self.vertices[vertex - 1] for vertex in face]
                corners.append(triangle)
            elif len(face) > 3:
                for i in range(1, len(face) - 1):
                    triangle = [self.vertices[face[0] - 1], self.vertices[face[i] - 1], self.vertices[face[i + 1] - 1]]
                    corners.append(triangle)

        return corners

    def get_normals(self):
        normals = []
        for face in self.triangle_mesh:
            normal = self.calculate_normal(self.vertices, face)
            normals.append(normal)

        return normals
    
    def get_centers(self):
        centers = []

        for face in self.triangle_mesh:
            sum = [0, 0, 0]
            for vertex in face:
                sum[0] += self.vertices[vertex][0]
                sum[1] += self.vertices[vertex][1]
                sum[2] += self.vertices[vertex][2]
            center = [coord / 3 for coord in sum]
            centers.append(center)

        return centers
    
    def get_neighbors(self):

        # Utwórz triangulację Delaunay
        triangulation = Delaunay(self.vertices)

        # Uzyskaj listę sąsiadów dla każdego wierzchołka
        neighbors = [list(set(triangulation.vertices.flatten()) - {i}) for i in range(len(self.vertices))]
        return neighbors
    '''
        faces_contain_this_vertex = [set([]) for _ in range(len(self.vertices))]
        for i in range(len(self.faces)):
            [v1, v2, v3] = self.faces[i]
            x1, y1, z1 = self.vertices[v1]
            x2, y2, z2 = self.vertices[v2]
            x3, y3, z3 = self.vertices[v3]
            faces_contain_this_vertex[v1].add(i)
            faces_contain_this_vertex[v2].add(i)
            faces_contain_this_vertex[v3].add(i)

        neighbors = []
        for i in range(len(self.faces)):
            [v1, v2, v3] = self.faces[i]
                
            n1 = self.find_neighbor(faces_contain_this_vertex, v1, v2, i)
            n2 = self.find_neighbor(faces_contain_this_vertex, v2, v3, i)
            n3 = self.find_neighbor(faces_contain_this_vertex, v3, v1, i)
                
            neighbors.append([n1, n2, n3])
        return neighbors
    def find_neighbor(self, faces_contain_this_vertex, vf1, vf2, except_face):
        for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
            if i != except_face:
                face = self.faces[i].tolist()
                face.remove(vf1)
                face.remove(vf2)
                return i

        return except_face
    '''
    def calculate_normal(self, vertices, triangle):
        edge1 = vertices[triangle[1]] - vertices[triangle[0]]
        edge2 = vertices[triangle[2]] - vertices[triangle[0]]

        cross_product = np.cross(edge1, edge2)

        normal_length = np.linalg.norm(cross_product)
        if normal_length != 0:
            normal = cross_product / normal_length
        else:
            normal = cross_product / 1.0

        return normal  

    def downsample(self):
        if len(self.vertices) <= self.max_vertices:
            print("downsampling not needed")
            return

        clusters = np.arange(len(self.vertices))  # Inicjalizacja klastrów

        while len(clusters) > self.max_vertices:
            min_distance = float('inf')
            cluster1_idx = -1
            cluster2_idx = -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    cluster1_indices = clusters[i]
                    cluster2_indices = clusters[j]
                    
                    cluster1 = np.mean(self.vertices[cluster1_indices], axis=0)
                    cluster2 = np.mean(self.vertices[cluster2_indices], axis=0)
                    
                    dist = np.linalg.norm(cluster1 - cluster2)

                    if dist < min_distance:
                        min_distance = dist
                        cluster1_idx = i
                        cluster2_idx = j

            merged_cluster = np.concatenate((clusters[cluster1_idx], clusters[cluster2_idx]))
            clusters = np.delete(clusters, [cluster1_idx, cluster2_idx])
            clusters = np.append(clusters, [merged_cluster])

        new_vertices = []

        for cluster_indices in clusters:
            cluster_vertices = self.vertices[cluster_indices]
            cluster_center = np.mean(cluster_vertices, axis=0)
            new_vertices.append(cluster_center)

        self.vertices = np.array(new_vertices)

        print(f"Downsampling done, new mesh size: {len(self.vertices)}")

def prepare_data(csv_file):
    parent_folder = Path(__file__).resolve().parent

    dataset = pd.read_csv(csv_file)

    for subset in ['train', 'test', 'val']:
        dataset_folder = parent_folder / 'dataset' / subset
        meshes_folder = dataset_folder / 'meshes'
        meshes_folder.mkdir(parents=True, exist_ok=True)
        numpy_folder = dataset_folder / 'data'
        numpy_folder.mkdir(parents=True, exist_ok=True)

        summary_csv_file = dataset_folder / 'summary.csv'
        dataset_subset = dataset[dataset['set'] == subset]
        dataset_subset = dataset_subset.drop(columns=['set'])
        dataset_subset.to_csv(summary_csv_file, index=False)

        print(f"Preparing {subset} dataset...")

        for index, row in dataset_subset.iterrows():
            start = time.time()
            print("id: ", row['id'], "link: ", row['link'])
            filename = "mesh_" + str(row['id']) + ".stl"
            mesh_file = meshes_folder / filename
            json_filename = str(row['id']) + '.json'
            json_file = dataset_folder / json_filename

            if json_file.is_file():
                print(f"Plik JSON o nazwie {json_filename} już istnieje. Pomijam dane o id: {row['id']}")
                continue

            try:
                download_file(row['link'], mesh_file)
                mesh = Mesh(mesh_file, CONFIG['max_vertices'])
                mesh_folder = numpy_folder / str(row['id'])
                mesh_folder.mkdir(parents=True, exist_ok=True)
                np.save(mesh_folder / 'vertices.npy', mesh.get_vertices())
                np.save(mesh_folder / 'faces.npy', mesh.get_faces())
                np.save(mesh_folder / 'triangles.npy', mesh.get_triangulated_faces())
                np.save(mesh_folder / 'corners.npy', mesh.get_corners())
                np.save(mesh_folder / 'centers.npy', mesh.get_centers())
                np.save(mesh_folder / 'normals.npy', mesh.get_normals())
                np.save(mesh_folder / 'neighbors.npy', mesh.get_neighbors())

                # Zapisz plik JSON tylko jeśli powiodło się pobieranie i przetwarzanie danych
                json_content = {
                    "name": filename,
                    "class": row['item'],
                    "link": row['link'],
                    "vertices": str(mesh_folder / 'vertices.npy'),
                    "faces": str(mesh_folder / 'faces.npy'),
                    "triangles": str(mesh_folder / 'triangles.npy'),
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
            except (IndexError, AttributeError, ValueError) as e:
                print(f"Błąd w przetwarzaniu danych: {e}. Pominięto dane o id: {row['id']}")
            stop = time.time()
            print("Czas operacji: ", stop - start)


if __name__ == "__main__":
    CSV = 'dataset.csv'
    prepare_data(CSV)
    print("corners: ")
    print(np.load("corners.npy"))