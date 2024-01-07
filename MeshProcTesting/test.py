from mesh_cutting import cut_mesh_growth_based, cut_mesh_shape_based
import pathlib
import numpy as np
import pandas as pd
import random
import time
import json
import os

#TODO:
#2. refactoring kodu

def run_defect_generation(input_path : pathlib.Path, output_path : pathlib.Path, cases_to_generate : int, **config):
  """
  Config should be a dictionary including:
  1) Dict: Number of holes with a given probability, e.g. {1: 0.75, 2: 0.2, 3: 0.05}
  2) Dict: Generation method with a given probability, e.g. {"growth": 0.3, "ball": 0.3, "ellipsoid": 0.3, "hiperboloid": 0.1} #ETC
  3) Dict: Parameter range for each method, e.g. {"growth" : dict_with_params, "ball" : other_dict_with_params} # ETC
  4) TO DO: csv opisujący dataset
  """
  
  #Load path
  input_path = pathlib.Path(input_path).resolve()
  output_path = pathlib.Path(output_path).resolve()

  train_path = input_path / "train"
  test_path = input_path / "test"
  val_path = input_path / "val"

  train_output_path = output_path / "train" / "meshes"
  test_output_path = output_path / "test" / "meshes"
  val_output_path = output_path / "val" / "meshes"

  config_output_path = output_path / "configs"
  if not os.path.exists(config_output_path):
    os.makedirs(config_output_path)

  train_summary = pd.read_csv(train_path / "summary.csv")
  test_summary = pd.read_csv(test_path / "summary.csv")
  val_summary = pd.read_csv(val_path / "summary.csv")

  for case in range(cases_to_generate):

    #input_subset = random.choice(["train", "test", "val"])
    input_subset = "val"
    if input_subset == "train":
      input_summary = train_summary
      input_path = train_path
      output_path = train_output_path
    elif input_subset == "test":
      input_summary = test_summary
      input_path = test_path
      output_path = test_output_path
    elif input_subset == "val":
      input_summary = val_summary
      input_path = val_path
      output_path = val_output_path
    #random item
    random_index = random.choice(input_summary.index)
    current_mesh_data = input_summary.iloc[random_index]
    mesh_id = str(current_mesh_data["id"])
    input_mesh_path = input_path / "data" / mesh_id
    mesh_features = os.listdir(input_mesh_path)
    #loading features
    current_mesh_features = {}
    for feature in mesh_features:
      current_mesh_features[feature] = np.load(input_mesh_path / feature)
    json_data_id = mesh_id + ".json"
    json_path = input_path / json_data_id
    with open(json_path) as json_path:
      data_json = json.load(json_path) #data_json to sparsowany json mesha
    print(f"Generating defect for mesh {input_mesh_path}")
    output_mesh_path = output_path / mesh_id
    current_config_params = define_from_distribution(config)
    num_defects = len(current_config_params)
    deleted_faces = []
    for i in range(num_defects):
      current_defect_config = current_config_params[f"{i}"]
      new_neighbors, new_faces, deleted = run_single_generation(input_mesh_path, output_mesh_path, current_mesh_features, **current_defect_config)
      current_mesh_features["neighbors.npy"] = new_neighbors
      current_mesh_features["faces.npy"] = new_faces
      deleted_faces.extend(deleted)
    output_file_cut = output_path / (mesh_id + "_" + str(case) + "_cut" + ".stl")
    output_file_fragment = output_path / (mesh_id + "_" + str(case) + "_fragment" + ".stl")
    write_stl(current_mesh_features["vertices.npy"], new_faces, current_mesh_features["normals.npy"], output_file_cut)
    write_stl(current_mesh_features["vertices.npy"], deleted_faces, current_mesh_features["normals.npy"], output_file_fragment)

    json_config_params = json.dumps(current_config_params)

    config_filename = mesh_id+"_"+str(case)+"config.json"
    with open(config_output_path / config_filename, "w") as config_json:
      config_json.write(json_config_params)
  
def run_single_generation(input_mesh_path, output_mesh_path, mesh_features, **params):
    
  faces = mesh_features["faces.npy"]
  centers = mesh_features["centers.npy"]
  neighbors = mesh_features["neighbors.npy"]
  method = params["method"]

  start_face = random.randint(0, len(faces) - 1)

  params.update({"start_face": start_face})

  if method == "growth":
    hole_size = params["hole_size"]
    if(hole_size) > len(faces):
      print("Invalid hole size for current mesh")
      return
    neighbors, faces, deleted = cut_mesh_growth_based(neighbors, faces, start_face, hole_size)
    for i in range(len(neighbors)):
      while len(neighbors[i]) < 3:
        neighbors[i] = np.append(neighbors[i], i)
    return neighbors, faces, deleted
  elif method == "sphere":
    neighbors, faces, deleted = cut_mesh_shape_based(method, faces, centers, neighbors, start_face, params)
    for i in range(len(neighbors)):
      while len(neighbors[i]) < 3:
        neighbors[i] = np.append(neighbors[i], i)
    return neighbors, faces, deleted
  elif method == "elipsoid":
    neighbors, faces, deleted = cut_mesh_shape_based(method, faces, centers, neighbors, start_face, params)
    for i in range(len(neighbors)):
      while len(neighbors[i]) < 3:
        neighbors[i] = np.append(neighbors[i], i)
    return neighbors, faces, deleted
  elif method == "hiperboloid":
    neighbors, faces, deleted = cut_mesh_shape_based(method, faces, centers, neighbors, start_face, params)
    for i in range(len(neighbors)):
      while len(neighbors[i]) < 3:
        neighbors[i] = np.append(neighbors[i], i)
    return neighbors, faces, deleted
  else:
    print("Invalid method")
    return

def random_sample_method(config):
  method = random.choices(list(config.keys()), weights = list(config.values()))
  return method
    
def define_from_distribution(params):

  output_params = {}

  hole_number = random.choices(list(params["number_of_holes"].keys()), weights = list(params["number_of_holes"].values()))
  for i in range(hole_number[0]):
    method = random.choices(list(params["generation_method"].keys()), weights=list(params["generation_method"].values()))[0]
    start_face = 0
    if method == 'growth':
      method_config = params["generation_parameters"][method]
      min_faces = method_config["min_faces"]
      max_faces = method_config["max_faces"]
      faces_number = random.randint(min_faces, max_faces)
      generated_params = {"method": method, "hole_size": faces_number, "start_face": start_face}
      output_params[f"{i}"] = generated_params
    
    elif method == 'sphere':
      method_config = params["generation_parameters"][method]
      min_radius = method_config["min_radius"]
      max_radius = method_config["max_radius"]
      radius = random.uniform(min_radius, max_radius)
      generated_params = {"method": method, "radius": radius, "start_face": start_face}
      output_params[f"{i}"] = generated_params

    elif method == 'elipsoid' or method == 'hiperboloid':
      method_config = params["generation_parameters"][method]
      a_axis_min = method_config["a-axis_min"]
      a_axis_max = method_config["a-axis_max"]
      b_axis_min = method_config["b-axis_min"]
      b_axis_max = method_config["b-axis_max"]
      c_axis_min = method_config["c-axis_min"]
      c_axis_max = method_config["c-axis_max"]
      a_axis = random.uniform(a_axis_min, a_axis_max)
      b_axis = random.uniform(b_axis_min, b_axis_max)
      c_axis = random.uniform(c_axis_min, c_axis_max)
      generated_params = {"method": method, "a_axis": a_axis, "b_axis": b_axis, "c_axis": c_axis, "start_face": start_face}
      output_params[f"{i}"] = generated_params
    
    else:
      print("Invalid method")

  return output_params

def write_stl(vertices, faces, normals, filename):
    with open(filename, 'w') as f:
        # Nagłówek pliku STL
        f.write("solid " + str(filename).split('/')[-1].split('.')[0] + "\n")

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

    print(f'STL file generated: {filename}')


config = {
  "number_of_holes" : {1: 0.75, 2: 0.2, 3: 0.05},
  "generation_method" : {"growth": 0.2, "sphere": 0.1, "elipsoid": 0.2, "hiperboloid": 0.5},
  "generation_parameters" : {"growth" : {"min_faces": 50, "max_faces": 100}, "sphere" : {"min_radius": 10, "max_radius": 50}, "elipsoid": {"a-axis_min": 10, "a-axis_max": 50, "b-axis_min": 10, "b-axis_max": 50, "c-axis_min": 10, "c-axis_max": 50}, "hiperboloid": {"a-axis_min": 0.5, "a-axis_max": 15, "b-axis_min": 0.5, "b-axis_max": 10, "c-axis_min": 0.5, "c-axis_max": 10}}
}

if __name__ == "__main__":

  start = time.time()

  run_defect_generation(input_path = pathlib.Path("./dataset"), output_path = pathlib.Path("./modified_dataset"), cases_to_generate = 5, **config)

  end = time.time()

  print(f"Time elapsed: {end - start}")