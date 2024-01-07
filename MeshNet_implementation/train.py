import torch
import data
from models import MeshNet
import time


def main():

    EPOCHS = 2

    dataset, class_to_id_map, path = data.load_dataset("train")
    dataset = data.MeshDataset(path, dataset, class_to_id_map)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False) #pin_memory na true po wrzuceniu tensorów do GPU

    config = {

    "structural_descriptor":
        {"num_kernel": 64, "sigma": 0.2},
    "mesh_convolution": {"aggregation_method": 'Concat'},
    "mask_ratio": 0.95,
    "dropout": 0.5,
    "num_classes": len(class_to_id_map)

    }

    model = MeshNet(cfg = config)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.00
    running_loss = 0.0
    running_corrects = 0

    for epoch in range(EPOCHS):

        start = time.time()

        print('-' * 60)
        print('Epoch: {} / {}'.format(epoch, EPOCHS))
        print('-' * 60)

        for i, (centers, corners, normals, neighbor_index, targets) in enumerate(dataloader):

            outputs = model(centers, corners, normals, neighbor_index)

            _, preds = torch.max(outputs, 1)

            print('preds: ', preds)
            print('targets: ', targets)

            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * centers.size(0)
            running_corrects += torch.sum(preds == targets.data)
        
        epoch_loss = running_loss / len(dataset)
        epoch_acc = running_corrects.double() / len(dataset)

        stop = time.time()

        epoch_time = stop - start

        print('Loss: {:.4f} Acc: {:.4f} Time: {:.2f}'.format(epoch_loss, epoch_acc, epoch_time))
    
    print('Best val acc: {:.4f}'.format(best_acc))


if __name__ == "__main__":
    main()
