# Deepimplant project

## data_prep:

- load_data.py takes txt file downloaded from https://medshapenet.ikim.nrw (MedShapeNet.txt)
- it creates two files: MedShapeNet.csv, which is txt file translated to csv, also added id and item_name columns for learning
- MSN-statistics.csv is a csv containing all category names and amounts of models in each category

## model-to-tensor:

- sample-model.py is a script that manually generates a 1x1x1 cube in .stl file format. It contains some description of the process to understand how does it work.
- TBA: script to convert a stl model to pytorch tensor and vice versa. Also will add some library test to check the efficiency of different stl processing libraries: trimesh, pytorch3D, open3D, maybe meshio