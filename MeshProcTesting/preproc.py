from pathlib import Path
import os
import pandas as pd
import requests
import numpy as np
from stl import mesh
import json

CSV_PATH = "dataset.csv"

def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Pobrano i zapisano plik jako {save_path}")
    else:
        print(f"Wystąpił błąd podczas pobierania pliku {save_path}")

def process_mesh(row, dataset_folder):
    filename = f"mesh_{row['id']}.stl"
    mesh_file = dataset_folder / 'meshes' / filename
    json_filename = f"{row['id']}.json"
    json_file = dataset_folder / json_filename

    if json_file.is_file():
        print(f"Plik JSON o nazwie {json_filename} już istnieje. Pomijam dane o id: {row['id']}")
        return

    try:
        download_file(row['link'], mesh_file)
        loaded_mesh = mesh.Mesh.from_file(mesh_file)

        mesh_folder = dataset_folder / 'data' / str(row['id'])
        mesh_folder.mkdir(parents=True, exist_ok=True)

        vertices = loaded_mesh.points
        faces = loaded_mesh.vectors.reshape(-1, 3)
        normals = loaded_mesh.normals

        # get neighbors
        faces_contain_this_vertex = []
        for i in range(len(vertices)):
            faces_contain_this_vertex.append(set([]))
        centers = []
        corners = []
        for i in range(len(faces)):
            print(f"faces[i]: {faces[i]}")
            v1, v2, v3 = faces[i]
            print(f"v1: {v1}, v2: {v2}, v3: {v3}")
            v1, v2, v3 = int(v1), int(v2), int(v3)
            print(f"v1: {v1}, v2: {v2}, v3: {v3}")
            x1, y1, z1 = vertices[v1]
            print(f"x1: {x1}, y1: {y1}, z1: {z1}")
            x2, y2, z2 = vertices[v2]
            x3, y3, z3 = vertices[v3]
            centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])
            corners.append([x1, y1, z1, x2, y2, z2, x3, y3, z3])
            faces_contain_this_vertex[v1].add(i)
            faces_contain_this_vertex[v2].add(i)
            faces_contain_this_vertex[v3].add(i)

        neighbors = []
        for i in range(len(faces)):
            print(f"faces[i]: {faces[i]}")
            v1, v2, v3 = faces[i]
            print(f"v1: {v1}, v2: {v2}, v3: {v3}")
            n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)
            n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
            n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
            neighbors.append([n1, n2, n3])

        centers = np.array(centers)
        corners = np.array(corners)


        np.save(mesh_folder / 'vertices.npy', vertices)
        np.save(mesh_folder / 'faces.npy', faces)
        np.save(mesh_folder / 'normals.npy', normals)
        np.save(mesh_folder / 'centers.npy', centers)
        np.save(mesh_folder / 'corners.npy', corners)
        np.save(mesh_folder / 'neighbors.npy', neighbors)

        json_content = {
            "name": filename,
            "class": row['item'],
            "link": row['link'],
            "vertices": str(mesh_folder / 'vertices.npy'),
            "faces": str(mesh_folder / 'faces.npy'),
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

def main():
    parent_folder = Path(__file__).parent.absolute()
    dataset = pd.read_csv(CSV_PATH)

    for subset in ['train', 'test', 'val']:
        dataset_folder = parent_folder / 'dataset' / subset
        dataset_folder.mkdir(parents=True, exist_ok=True)

        meshes_folder = dataset_folder / 'meshes'
        meshes_folder.mkdir(parents=True, exist_ok=True)

        for index, row in dataset[dataset['set'] == subset].iterrows():
            print(f"Preparing {subset} dataset - id: {row['id']}, link: {row['link']}")
            process_mesh(row, dataset_folder)

        summary_csv_file = dataset_folder / 'summary.csv'
        dataset_subset = dataset[dataset['set'] == subset].drop(columns=['set'])
        dataset_subset.to_csv(summary_csv_file, index=False)
        print(f"Summary CSV file saved: {summary_csv_file}")

if __name__ == "__main__":
    main()
