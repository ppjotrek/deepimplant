import numpy as np

path = "./vertices.npy"

vtcs = np.load(path)
print(len(vtcs))
print(vtcs[0])