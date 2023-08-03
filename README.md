# Medical imaging mesh processing

### data_prep:

- load_data.py takes txt file downloaded from https://medshapenet.ikim.nrw (MedShapeNet.txt)
- it creates two files: MedShapeNet.csv, which is txt file translated to csv, also added id and item_name columns for learning
- MSN-statistics.csv is a csv containing all category names and amounts of models in each category

### object-tensor-interface:

- **generate-sample-stl.py** is a script that manually generates a 1x1x1 cube in .stl file format. It contains some description of the process to understand how does it work.
- **trimesh-utils.py** is a script that can be used as interface for loading any stl or obj file as a pytorch script. You can find more about it in examples.
