import numpy as np

def cut_mesh_growth_based(neighbors, faces, start_face, hole_size):

    faces_copy = np.copy(faces)

    #check if start_face is valid (if it's not out of range)
    if start_face < 0 or start_face >= len(faces_copy):
        print("Błędny indeks startowy.")
        return neighbors, faces_copy, []
    
    visited_faces = set()
    faces_to_remove = []

    queue = [start_face]

    while queue and len(faces_to_remove) < hole_size:
        current_face = queue.pop(0)
        if current_face not in visited_faces:
            visited_faces.add(current_face)
            faces_to_remove.append(current_face)
            #print(current_face)
            for neighbor_face in neighbors[current_face]:
                if neighbor_face not in visited_faces:
                    queue.append(neighbor_face)

    #update faces and normals
    faces_copy = np.delete(faces_copy, faces_to_remove, axis=0)

    #update neighbors
    neighbors_copy = [np.setdiff1d(neighbor, faces_to_remove) for neighbor in neighbors]

    deleted_faces = faces[faces_to_remove]

    return neighbors_copy, faces_copy, deleted_faces