import data
import torch
import os
import time
from models import MeshNet

dataset, class_to_id_map, path = data.load_dataset(subset = 'val')
Dataset = data.MeshDataset(path, dataset, class_to_id_map)
data_loader = torch.utils.data.DataLoader(Dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)

config = {

    "structural_descriptor":
        {"num_kernel": 64, "sigma": 0.2},
    "mesh_convolution": {"aggregation_method": 'Concat'},
    "mask_ratio": 0.95,
    "dropout": 0.5,
    "num_classes": len(class_to_id_map)

}

model = MeshNet(cfg = config, require_fea=True)

for i, (centers, corners, normals, neighbor_index, targets) in enumerate(data_loader):
    centers = centers
    print(centers.shape)
    corners = corners
    print(corners.shape)
    normals = normals
    print(normals.shape)
    neighbor_index = neighbor_index
    print(neighbor_index.shape)
    targets = targets

    start = time.time()

    feas = model(centers, corners, normals, neighbor_index)

    stop = time.time()

    print("Feas size: ", feas.shape)

    print("Propagation time: ", stop - start)
    print("Mesh size: ", centers.shape[2])

    break