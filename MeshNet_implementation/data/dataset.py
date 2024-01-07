import numpy as np
import torch
import pandas as pd
from pathlib import Path
import json

def load_dataset(subset, filename: str = "summary.csv"):
    file_path = Path(__file__)
    parent_path = Path.resolve(file_path.parent.parent)
    dataset_path = parent_path / "dataset"
    dataset_path = dataset_path / subset


    dataset = pd.read_csv(dataset_path / filename)
    labels = dataset["item"].unique()

    class_to_index_map = {}

    for index, item in enumerate(labels):
        if pd.notna(item):
            class_to_index_map[item] = index

    class_to_index_map_file = str(parent_path / "data" / "class_to_index_map.py")

    with open(class_to_index_map_file, "w") as f:
        f.write("class_to_index_map = " + str(class_to_index_map))
    
    return dataset, class_to_index_map, dataset_path


class MeshDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, dataset, class_to_index_map):
        self.root = data_path
        self.dataset = dataset
        self.class_to_index_map = class_to_index_map
        self.data = []
        for index, row in dataset.iterrows():
            entry_name = str(row["id"]) + ".json"
            data_file = self.root / entry_name
            if data_file.exists():
                self.data.append((data_file, self.class_to_index_map[row["item"]]))
            else:
                print("File {} not found!".format(data_file))

    def __getitem__(self, index):
        path, label = self.data[index]
        entry_content = json.load(open(path, "r"))
        target = torch.tensor(label, dtype=torch.long)
        face = torch.from_numpy(np.load(entry_content["faces"])).float()
        centers = torch.from_numpy(np.load(entry_content["centers"])).float()
        corners = torch.from_numpy(np.load(entry_content["corners"])).float()
        normals = torch.from_numpy(np.load(entry_content["normals"])).float()
        neighbors = torch.from_numpy(np.load(entry_content["neighbors"])).long()
        face = face.permute(1, 0).contiguous()
        centers = centers.permute(1, 0).contiguous()
        corners = corners.reshape(9, -1)
        normals = normals.permute(1, 0).contiguous()
        return centers, corners, normals, neighbors, target

    def __len__(self):
        return len(self.data)