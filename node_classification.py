###
# 1. read pcl and labels
# 2. create graph
# 3. compute edges (T.Knn)
# 4. create GCN model
# 5. train on labeled nodes (or estimate some and train on those)
###
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import open3d as o3d
import os
import numpy as np
import pdb
import json
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
from torch_geometric.nn import GCNConv
#from torch.nn import Linear
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, input_features, hidden_channels, output_classes):
        super().__init__()

        self.conv1 = GCNConv(input_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels//2)
        self.conv3 = GCNConv(hidden_channels//2, output_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x

def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    # Compute the loss solely based on the training nodes.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def train_with_loader(train_loader):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         data.to(device)
         out = model(data.x, data.edge_index)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

    return loss

def test_with_loader(loader):
     model.eval()

     correct = 0
     correct_ratio = 0
     correct_areas = [0,0]
     for data in loader:  # Iterate in batches over the training/test dataset.
         data.to(device)
         out = model(data.x, data.edge_index)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         np_labels = data.y.cpu().numpy()
         correct_preds = pred.cpu().numpy() == np_labels
         correct += np.sum(correct_preds)  # Check against ground-truth labels.
         data_count = np_labels.shape[0]
         count_0 = np.sum(np_labels < 0.5)
         count_1 = np.sum(np_labels > 0.5)
         correct_areas[0] += np.sum(correct_preds[np_labels < 0.5]) / count_0
         correct_areas[1] += np.sum(correct_preds[np_labels > 0.5]) / count_1
         correct_ratio += np.sum(correct_preds) / data_count
     correct /= len(loader.dataset)
     correct_areas = np.asarray(correct_areas) / len(loader.dataset)
     correct_ratio /= len(loader.dataset)

     return correct, correct_areas, correct_ratio # Derive ratio of correct predictions.

def test(data):
    model.eval()
    data.to(device)
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    # Check against ground-truth labels.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    # Derive ratio of correct predictions.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return pred, test_acc

def predict(data):
    model.eval()
    data.to(device)
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    return pred.cpu().numpy()

def prepare_dataset():

    dataset = []
    pcls = []
    names = []

    # 1. read pcl and labels
    group = 15
    RePAIR_dataset = "/home/palma/Unive/RePAIR/Datasets/RePAIR_dataset"
    #RePAIR_dataset = "/Users/Palma/Documents/Projects/Unive/RePAIR/Datasets/RePAIR_dataset"
    group_folder = os.path.join(RePAIR_dataset, f"group_{group}")
    processed_folder = os.path.join(group_folder, "processed")
    labeled_folder = os.path.join(group_folder, "labeled")
    labels_folder = os.path.join(os.getcwd(), "labels")
    output_folder = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    json_path = os.path.join(labels_folder, f"Group_{group}-v0.2.json")
    #pdb.set_trace()
    with open(json_path, 'r') as j:
        contents = json.loads(j.read())
    # samples which have been labeled
    samples_data = contents['dataset']['samples']
    for sample in samples_data:
        # check which piece is
        s_name = sample['name']
        rpf_name = s_name[:9]
        #pdb.set_trace()
        print("found", rpf_name)
        # get labels
        if sample['labels']['ground-truth'] and rpf_name[:3] == 'RPf':
            print("Working on", rpf_name)
            s_labels = np.asarray(
                sample['labels']['ground-truth']['attributes']['point_annotations'])
            # read original pcl
            orig_path = os.path.join(processed_folder, f"{rpf_name}.ply")
            orig_pcl = o3d.io.read_point_cloud(orig_path)
            assert(len(orig_pcl.points) == len(s_labels)), "problem"

            # pdb.set_trace()
            # 2. crate a pytorch data object
            x = np.zeros((len(np.asarray(orig_pcl.points)), 6))
            x[:, 0:3] = np.asarray(orig_pcl.points)
            x[:, 3:6] = np.asarray(orig_pcl.normals)
            data = Data(x=torch.tensor(x, dtype=torch.float32),
                        edge_index=None,
                        edge_attr=None,
                        y=torch.tensor(s_labels),
                        pos=torch.tensor(np.asarray(
                                orig_pcl.points), dtype=torch.float32),
                        train_mask=torch.tensor(np.ones_like(
                            np.asarray(orig_pcl.points)[:, 0]), dtype=torch.bool),
                        val_mask=torch.tensor(np.ones_like(np.asarray(
                            orig_pcl.points)[:, 0]), dtype=torch.bool),
                        test_mask=torch.tensor(np.ones_like(np.asarray(orig_pcl.points)[:, 0]), dtype=torch.bool))
            # 3. compute edges (T.Knn)
            edge_creator = T.KNNGraph(k=5)
            edge_creator(data)
            dataset.append(data)
            pcls.append(orig_pcl)
            names.append(rpf_name)
    return dataset, pcls, names


def show_results(pred, pcl):
    colors = np.zeros((pred.shape[0], 3))
    colors[pred == 0] = [0, 255, 0]
    colors[pred == 1] = [255, 0, 0]
    pcl.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcl])

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    #pdb.set_trace()
    # get the 6 labeled pcls as graph (pytorch data object)
    dataset, pcls, names = prepare_dataset()

    # prepare the model
    input_features = 6
    hidden_channels = 32
    output_classes = 2
    print(f"GCN Model with: \
          {input_features} input features, \
          {hidden_channels} hidden_channels and \
          {output_classes} output_classes")
    # 4. create GCN model
    model = GCN(input_features=input_features,
                hidden_channels=hidden_channels,
                output_classes=output_classes)
    model.to(device)

    # 5. train on labeled nodes (or estimate some and train on those)
    model.eval()
    #out = model(data.x, data.edge_index)
    # visualize(out, color=data.y)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print("start training..")
    EPOCHS = 1201
    test_acc = 0.0
    acc_intact = 0.0
    acc_broken = 0.0

    train_test_split = 3
    train_dataset = dataset[:train_test_split]
    test_dataset = dataset[train_test_split:]
    train_files = names[:train_test_split]
    test_files = names[train_test_split:]
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    for epoch in range(1, EPOCHS):
        loss = train_with_loader(train_loader)
        if epoch % 10 == 0:
            #pdb.set_trace()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        if epoch % 100 == 0:
            #pdb.set_trace()
            train_avg_correct_points, train_acc_areas, train_acc_ratio = test_with_loader(train_loader)
            test_avg_correct_points, test_acc_areas, test_acc_ratio = test_with_loader(test_loader)
            print('Training:')
            print(f'average correct points: {train_avg_correct_points:.4f}, accuracy ratio: {train_acc_ratio:.4f}')
            print(f'accuracy on intact surfaces: {train_acc_areas[0]:.4f}, accuracy on broken surfaces: {train_acc_areas[1]:.4f}')
            print('Testing:')
            print(f'average correct points: {test_avg_correct_points:.4f}, accuracy ratio: {test_acc_ratio:.4f}')
            print(f'accuracy on intact surfaces: {test_acc_areas[0]:.4f}, accuracy on broken surfaces: {test_acc_areas[1]:.4f}')
    results = {
        'loss': loss.item(),
        'training': {
            'acc_ratio': train_acc_ratio,
            'acc_intact': train_acc_areas[0],
            'acc_broken': train_acc_areas[1],
            'files': train_files
        },
        'test': {
            'acc_ratio': test_acc_ratio,
            'acc_intact': test_acc_areas[0],
            'acc_broken': test_acc_areas[1],
            'files': test_files
        },
        'epochs': EPOCHS,
        'device': str(device)
    }
    output_folder = os.path.join(os.getcwd(), 'results_classification')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    with open(os.path.join(output_folder,
        f"results_after_{epoch}_epochs.json"), 'w') as jf:
        json.dump(results, jf, indent=3)
    # if test_acc > 0 and acc_broken > 0.5:
    #     np.savetxt(os.path.join(output_folder,
    #         f"pred_{rpf_name}_after_{epoch}_epochs.txt"), pred.numpy())


    for j in range(3, 6):
        pred = predict(dataset[j]) # pred returned is already .cpy().numpy()
        pcl = pcls[j]
        print(f"showing prediction for pcl {j}: {names[j]}")
        show_results(pred, pcl)
