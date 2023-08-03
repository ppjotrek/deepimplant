import torch
import trimesh
import numpy as np

def load_stl_to_tensor(path):
    mesh = trimesh.load(path)
    vertices = mesh.vertices
    faces = mesh.faces
    vertices = torch.from_numpy(vertices).float()
    faces = torch.from_numpy(faces).long()
    return vertices, faces

def load_obj_to_tensor(path):
    mesh = trimesh.load(path)
    vertices = mesh.vertices
    faces = mesh.faces
    vertices = torch.from_numpy(vertices).float()
    faces = torch.from_numpy(faces).long() - 1 # obj indexing start from 1
    return vertices, faces

def load_to_tensor(path, file_format):
    mesh = trimesh.load(path)
    vertices = mesh.vertices
    faces = mesh.faces
    vertices = torch.from_numpy(vertices).float()
    if(file_format == 'obj'):
        faces = torch.from_numpy(faces).long() - 1
    elif(file_format == 'stl'):
        faces = torch.from_numpy(faces).long()
    else:
        raise ValueError('file_format should be obj or stl')
    return vertices, faces

class Mesh_Tensor:

    def __init__(self, path):
        self.path = path
        self.vertices, self.faces = load_to_tensor(path, path.split('.')[-1])
    
    def get_vertices(self):
        return self.vertices
    
    def get_faces(self):
        return self.faces
    
    def get_path(self):
        return self.path
    
    def get_mesh(self):
        return trimesh.load(self.path)
    
    def get_stl_like(self):
        stl_like = []

        for face in self.faces:
            triangle = []
            for vertex_index in face:
                triangle.append(self.vertices[vertex_index].tolist())
            stl_like.append(triangle)
    
        stl_like_structure = np.array(stl_like, dtype=np.float32)
        tensor = torch.tensor(stl_like_structure, dtype = torch.float32)
        return tensor
    
    def get_centers(self):
        mesh = self.get_mesh()
        centers = mesh.vertices[mesh.faces].mean(axis=1)
        return torch.from_numpy(centers).float()
    
    def get_normals(self):
        mesh = self.get_mesh()
        normals = mesh.face_normals
        return torch.from_numpy(normals).float()
    
    def get_corners(self):
        mesh = self.get_mesh()
        corners = mesh.vertices[mesh.faces]
        return torch.from_numpy(corners).float()
    
    def get_edges(self):
        mesh = self.get_mesh()
        edges = mesh.edges_unique
        return torch.from_numpy(edges).long()