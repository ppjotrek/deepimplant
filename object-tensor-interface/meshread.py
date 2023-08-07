import numpy as np
import torch

def read_obj_file(path: str):

    #Function to read .obj files and return vertices and faces as lists

    vertices = []
    faces = []
    texture_vertices = []

    with open(path, "r") as f:

        for line in f:
            if line.startswith('v'):
                line.split(' ')
                line = line[1:].split()
                for i in range(3):
                    line[i] = float(line[i])
                vertices.append(line)
            elif line.startswith('f'):
                line.split(' ')
                line = line[1:].split()
                for i in range(4):
                    line[i] = int(line[i])
                faces.append(line)
            elif line.startswith('vt'):
                pass
                #TODO: implement texture vertices
            elif line.startswith('#') or line.startswith(' ') or line.startswith('/n'):
                pass
            else:
                pass

    return vertices, faces


def read_stl_file(path: str, return_normals=False):

    #Function to read .stl files and return vertices and faces as lists. Also returns normals if return_normals=True

    vertices = []
    faces = []
    normals = []

    with open(path, "r") as f:

        for line in f:
            if line.strip().startswith('vertex'):
                vertex = [float(coord) for coord in line.strip().split()[1:]]
                vertices.append(vertex)
            elif line.strip().startswith('endloop'): #saving face after reading 3 vertices
                face = [len(vertices) - 3, len(vertices) - 2, len(vertices) - 1]
                faces.append(face)
            elif line.strip().startswith('facet normal'):
                normal = [float(coord) for coord in line.strip().split()[2:]]
                normals.append(normal)
    
    if return_normals:
        return vertices, faces, normals
    else:
        return vertices, faces
    

class Mesh:

    def __init__(self, path: str):
        if path.split('.')[-1] == 'obj':
            self.vertices, self.faces = read_obj_file(path)
            self.fileformat = 'obj'
        elif path.split('.')[-1] == 'stl':
            self.vertices, self.faces, self.normals = read_stl_file(path, return_normals=True)
            self.fileformat = 'stl'
        else:
            raise ValueError('Unsupported file format. Supported formats are .obj and .stl. Format you tried to use is: ', path.split('.')[-1])
        self.path = path
        self.triangle_mesh = self.triangulate_faces()
    
    def triangulate_faces(self):
        if self.fileformat == 'stl':
            return self.faces
        else:
            triangles = []
            for face in self.faces:
                # Sprawdzamy, czy to jest wielokąt (więcej niż 3 wierzchołki)
                if len(face) > 3:
                    # Tworzymy trójkąty przez dzielenie wielokąta na kolejne trójkąty
                    for i in range(1, len(face) - 1):
                        triangle = [face[0]-1, face[i]-1, face[i + 1]-1]
                        triangles.append(triangle)
                else:
                    # Jeśli to jest trójkąt, po prostu dodajemy go do listy trójkątów
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

        for face in self.faces:
            if len(face) == 3:
                # W przypadku trójkątów (STL) dodajemy je bezpośrednio do corners
                triangle = [self.vertices[int(vertex_index)] for vertex_index in face]
                corners.append(triangle)
            elif len(face) > 3:
                # W przypadku wielokątów (OBJ) dzielimy je na kolejne trójkąty
                for i in range(1, len(face) - 1):
                    triangle = [self.vertices[int(face[0]) -1], self.vertices[int(face[i]) -1], self.vertices[int(face[i + 1]) -1]] #-1, bo indeksowanie od 1 w obj
                    corners.append(triangle)

        return corners

    def get_normals(self):
        normals = []
        for face in self.triangle_mesh:
            normal = self.calculate_normal(self.vertices, face)
            normals.append(normal)
        return normals
    
    def get_centers(self): #TODO: zrobić to lepiej

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

    def calculate_normal(self, vertices, triangle):
        # Obliczamy wektor różnicy dla dwóch kolejnych punktów
        edge1 = [vertices[triangle[1]][j] - vertices[triangle[0]][j] for j in range(3)]
        edge2 = [vertices[triangle[2]][j] - vertices[triangle[0]][j] for j in range(3)]

        # Obliczamy iloczyn wektorowy dwóch krawędzi
        cross_product = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0]
        ]

        # Normalizujemy wektor normalny (zmniejszamy jego długość do 1)
        normal_length = (cross_product[0]**2 + cross_product[1]**2 + cross_product[2]**2)**0.5
        normal = [coord / normal_length for coord in cross_product]

        return normal