import os

line = ''
json_name = ''
parent_path = './'
dir_path = os.path.join(parent_path, 'Dataset')
with open(os.path.join(dir_path, 'dataset.csv'), 'w') as csv:
    csv.write('filename,file_link,class,mesh_link')

with open(os.path.join('/home/ppjotrek/Python/Medshapenet/data_prep/MedShapeNetDataset.csv'), 'r') as file:
    for entry in file:
        entry = entry.split(',')
        if entry[0] == 'id':
            continue
        else:
            json_name = 'mesh_' + entry[0] + '.json'
            json_path = os.path.join(dir_path, json_name)
            json = open(json_path, 'w')
            json.write("{\n")
            json_line = "    'category': " + entry[1] + ",\n"
            json.write(json_line)
            json.write("\n}")
            json.close()
            line = json_name + ',' + json_path + ',' + entry[1] + ',' + entry[2]
            with open(os.path.join(dir_path, 'dataset.csv'), 'a') as csv:
                csv.write(line)

print('Done creating files!')


#Pandas do obsługi CSV
#Pathlib zamiast os.path.join()
#Zabudować w funkcję i run if __filename__=='main'
#torch lightning - doczytać - do budowy trainera