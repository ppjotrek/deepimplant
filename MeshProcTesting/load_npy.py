import numpy as np

FOLDER = "0ddd75bb"
filename = "faces.npy"

corners = np.load(FOLDER + "/" + filename)
print(corners)