import numpy as np

def cut_mesh_shape_based(shape, faces, centers, neighbors, start_face, params):
    
    faces_copy = np.copy(faces)
    centers_copy = np.copy(centers)
    neighbors_copy = np.copy(neighbors)

    #check if start_face is valid (if it's not out of range)
    if start_face < 0 or start_face >= len(faces_copy):
        print("Invalid start index")
        return faces_copy, centers_copy, []
    
    faces_to_remove = []

    for i in range(len(faces)):
        if is_in_shape(shape, centers[i], centers[start_face], params):
            faces_to_remove.append(i)

    #update faces and normals
    faces_copy = np.delete(faces_copy, faces_to_remove, axis=0)

    #update neighbors
    neighbors_copy = [np.setdiff1d(neighbor, faces_to_remove) for neighbor in neighbors]
    deleted_faces = faces[faces_to_remove]

    return neighbors_copy, faces_copy, deleted_faces
    

def is_in_shape(shape, point, center, params):
    if(shape == 'sphere'):
        if ((point[0] - center[0])**2 + (point[1] - center[1])**2 + (point[2] - center[2])**2) <= (params["radius"]**2):
            return True
        else:
            return False
    elif(shape == 'elipsoid'):
        a_axis = params["a_axis"]
        b_axis = params["b_axis"]
        c_axis = params["c_axis"]
        if ((point[0] - center[0])**2)/(a_axis**2) + ((point[1] - center[1])**2)/(b_axis**2) + ((point[2] - center[2])**2)/(c_axis**2) <= 1:
            return True
        else:
            return False
    elif(shape == 'hiperboloid'):
        a_axis = params["a_axis"]
        b_axis = params["b_axis"]
        c_axis = params["c_axis"]
        if ((point[0] - center[0])**2)/(a_axis**2) + ((point[1] - center[1])**2)/(b_axis**2) - ((point[2] - center[2])**2)/(c_axis**2) <= 1:
            return True
        else:
            return False
    else:
        print("Invalid shape ", shape)
        return False