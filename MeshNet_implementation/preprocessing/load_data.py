import os
import uuid
import random
import csv

links = []
items = {}
sets_counts = {
    'train': 0,
    'test': 0,
    'val': 0
}

with open(os.path.join('MedShapeNetDataset.txt'), 'r') as f:
    for line in f:
        line = line.strip()
        links.append(line)

for link in links:
    if "nii.g_1.stl" in link:
        start_index = link.find('=s') + 2
        end_index = link.find('.nii.g_1.stl')
        item = link[start_index:end_index]
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
    
    #if full_name == 'heart' or full_name == 'liver' or full_name == 'colon' or full_name == 'brain' or full_name == 'skull_ct(part)' or full_name == 'vessel_tree' or full_name == 'sacrum' or full_name == 'face':
    items[link] = full_name

# Podział na zbiory treningowy, testowy i walidacyjny
random.shuffle(links)
total_samples = len(links)
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1

train_links = links[:int(total_samples * train_ratio)]
test_links = links[int(total_samples * train_ratio):int(total_samples * (train_ratio + test_ratio))]
val_links = links[int(total_samples * (train_ratio + test_ratio)):]

sets_counts['train'] = len(train_links)
sets_counts['test'] = len(test_links)
sets_counts['val'] = len(val_links)

# Tworzenie pliku dataset.csv
dataset_file = 'dataset.csv'
with open(dataset_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'item', 'link', 'set'])
    
    for link_set, set_name in [(train_links, 'train'), (test_links, 'test'), (val_links, 'val')]:
        for link in link_set:
            random_hash = str(uuid.uuid4())[:8]
            if link in items:
                writer.writerow([random_hash, items[link], link, set_name])

# Generowanie statystyk
stats_file = 'MSN-statistics.csv'
with open(stats_file, 'w') as f:
    cols = ['set', 'count']
    f.write(','.join(cols) + '\n')
    for set_name, count in sets_counts.items():
        f.write('{},{}\n'.format(set_name, count))

    unique = {}
    for item in items.values():
        if item in unique:
            unique[item] += 1
        else:
            unique[item] = 1
    for i, item in enumerate(unique):
        f.write('{},{},{}\n'.format(i, item, unique[item]))