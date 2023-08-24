import json
import numpy as np
from pathlib import Path

#Main folder path
parent_folder = Path(__file__).resolve().parent.parent
dataset_folder = parent_folder / 'dataset'

json_file = dataset_folder / "0.json"

with open(json_file) as f:
    data = json.load(f)
print(data)

vertices = np.load(data['vertices'])
print("Vertices:")
print(vertices)