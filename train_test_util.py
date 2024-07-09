import numpy as np 
import torch 
import open3d as o3d 
import matplotlib.pyplot as plt

def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    # Compute the loss solely based on the training nodes.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def training_loop_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         data.to(device)
         #breakpoint()
         out = model(data.x, data.edge_index)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

    return loss

def test_with_loader(model, device, loader):
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

def test(model, data):
    model.eval()
    data.to(device)
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    # Check against ground-truth labels.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    # Derive ratio of correct predictions.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return pred, test_acc

def predict(model, data, device):
    model.eval()
    data.to(device)
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    return pred.cpu().numpy()

def show_results(pred, pcl):
    cmap = plt.get_cmap('jet').resampled(5)
    #breakpoint()
    #colors = np.zeros((pred.shape[0], 3))
    colors = cmap(pred)
    #colors[pred == 0] = [0, 255, 0]
    #colors[pred == 1] = [255, 0, 0]
    pcl.colors = o3d.utility.Vector3dVector(colors[:,:3])
    o3d.visualization.draw_geometries([pcl])