import os
links = []
items = {}

with open(os.path.join('MedShapeNetDataset.txt'), 'r') as f:
    
    for line in f:
        line = line.strip()
        links.append(line)

for link in links:
    # Sprawdzamy czy link zawiera "nii.g_1.stl"
    if "nii.g_1.stl" in link:
        start_index = link.find('=s') + 2
        end_index = link.find('.nii.g_1.stl')
        item = link[start_index:end_index]
    # Sprawdzamy czy link zawiera ".stl"
    elif ".stl" in link:
        start_index = link.find('=') + 1
        end_index = link.find('.stl')
        item = link[start_index:end_index]
    else:
        item = 'Unknown'

    file_name = item.split('_')[1:]
    full_name = ''.join(file_name)
    if full_name[:4] == 'tree':
        full_name = 'vessel_tree'
    if full_name[:5] == 'skull':
        full_name = 'skull_ct(part)'
    if full_name == '1' or full_name == '2' or full_name == '3' or full_name == '4' or full_name == '5' or full_name == '6' or full_name == '7' or full_name == '8' or full_name == '9' or full_name == '10' or full_name == '11' or full_name == '12':
        full_name = 'face'
    items[link] = full_name

output_file = 'MedShapeNetDataset.csv'
with open(output_file, 'w') as f:
    cols = ['id', 'item', 'link']
    f.write(','.join(cols) + '\n')
    for i, link in enumerate(items):
        f.write('{},{},{}\n'.format(i, items[link], link))

stats_file = 'MSN-statistics.csv'
with open(stats_file, 'w') as f:
    cols = ['id', 'item', 'amount']
    f.write(','.join(cols) + '\n')
    unique = {}
    for item in items.values():
        if item in unique:
            unique[item] += 1
        else:
            unique[item] = 1
    for i, item in enumerate(unique):
        f.write('{},{},{}\n'.format(i, item, unique[item]))
