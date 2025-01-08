import numpy as np 
import torch 
import open3d as o3d 
import matplotlib.pyplot as plt
from network.gnns import GCN, GAT, recognitionGCN

def get_loss(cfg, weight):
    # LOSS
    if cfg['loss'] == "NLL":
        criterion = torch.nn.NLLLoss(weight=weight) #()
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=weight) #NLLLoss()
    return criterion

def get_weights(cfg):
    # WEIGHTS (for imbalanced datasets)
    if cfg['task'] == 'detection':
        weight = torch.tensor([1, cfg['weight_obj']], dtype=torch.float32)
    elif cfg['task'] == 'recognition':
        weights = np.ones((cfg['num_classes'])) * cfg['weight_obj']
        # weights[0] /= 5
        weight = torch.tensor(weights, dtype=torch.float32)
    
    return weight

def show_data(set, num_examples):
    examples_ids = np.random.choice(len(set), num_examples)
    # show data
    for k in examples_ids:
        pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(set[k].x[:,:3]))
        pcl.paint_uniform_color((0,0,1))
        name = f"example {k}: object of class {int(set[k].y.argmax(dim=1))}"
        o3d.visualization.draw_geometries([pcl], window_name = name)
    # breakpoint()

def print_parameters(cfg):
    print("#" * 50)
    print("# PARAMETERS")
    print("#" * 50)
    for cfg_key in cfg.keys():
        print(f"# {cfg_key}:{cfg[cfg_key]}")
    print("#" * 50)

def build_model(cfg):

    if cfg['model'] == 'GAT':
        model = GAT(input_features=cfg["input_features"],
                    hidden_channels=cfg['hidden_channels'],
                    output_classes=cfg['num_classes'])
    elif cfg['model'] == 'GCN':
        model = recognitionGCN(input_features=cfg['input_features'],
                            hidden_channels=cfg['hidden_channels'],
                            output_classes=cfg['num_classes'], 
                            dropout=cfg['dropout_rate'])
    else:
        print("WHICH MODEL?")

    print(f"{cfg['model']} Model with: \
          {cfg['input_features']} input features, \
          {cfg['hidden_channels']} hidden_channels and \
          {cfg['num_classes']} output_classes")
    return model

def add_noise(data, noise_strength):
    z_range = data.x[:,2].max().item() - data.x[:,2].min().item()
    data.x[:,2] += np.random.uniform(-1, 1, data.x.shape[0]) * noise_strength * z_range
    return data

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


def show_results(pred, pcl, window_name="results"):
    cmap = plt.get_cmap('jet').resampled(5)
    #breakpoint()
    #colors = np.zeros((pred.shape[0], 3))
    colors = cmap(pred)
    #colors[pred == 0] = [0, 255, 0]
    #colors[pred == 1] = [255, 0, 0]
    pcl.colors = o3d.utility.Vector3dVector(colors[:,:3])
    o3d.visualization.draw_geometries([pcl], window_name=window_name)
