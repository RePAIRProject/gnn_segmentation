import torch 
import numpy as np
import torch_geometric.transforms as T
import os 
from torch_geometric.data import Data
import open3d as o3d

def dataset_binary_pcl_labels(folder, k=25):

    pcl_folder = os.path.join(folder, 'pcl')
    pcls = os.listdir(pcl_folder)
    labels_folder = os.path.join(folder, 'labels')
    labels = os.listdir(labels_folder)
    pcls.sort(); labels.sort()
    assert len(pcls) == len(labels), "wrong number of labels/pcls"
    dataset = []
    print('\nLoading dataset..\n')
    for pcl_path, labels_path in zip(pcls, labels):
        print(f"Reading {pcl_path} and {labels_path}..", end="\r")
        pcl = o3d.io.read_point_cloud(os.path.join(pcl_folder, pcl_path))
        pos_np = np.asarray(pcl.points)
        pos = torch.tensor(pos_np, dtype=torch.float32)
        x = np.zeros((len(np.asarray(pcl.points)), 6))
        normals = pcl.normals
        colors = pcl.colors
        x[0:3] = pos_np
        x[3:6] = normals
        x[6:9] = colors
        xt = torch.tensor(xt, dtype=torch.float32)
        y_np = np.loadtxt(os.path.join(labels_folder, labels_path))
        if np.min(y_np) == 0:
            y_np[y_np==0] = 1
        y_np[y_np > 1] = 2
        y = torch.tensor(y_np, dtype=torch.long)
        data = Data(x=xt, y=y-1, pos=pos, edge_index=None, edge_attr=None)
        # 3. compute edges (T.Knn)
        edge_creator = T.KNNGraph(k=k)
        data = edge_creator(data)
        dataset.append(data)
    print('\nDataset loaded!')

    return dataset

def dataset_pcl_labels(folder, k=15):

    pcl_folder = os.path.join(folder, 'pcl')
    pcls = os.listdir(pcl_folder)
    labels_folder = os.path.join(folder, 'labels')
    labels = os.listdir(labels_folder)
    pcls.sort(); labels.sort()
    assert len(pcls) == len(labels), "wrong number of labels/pcls"
    dataset = []
    print('\nLoading dataset..\n')
    for pcl_path, labels_path in zip(pcls, labels):
        print(f"Reading {pcl_path} and {labels_path}..", end="\r")
        pcl = o3d.io.read_point_cloud(os.path.join(pcl_folder, pcl_path))
        pos_np = np.asarray(pcl.points)
        pos = torch.tensor(pos_np, dtype=torch.float32)
        y_np = np.loadtxt(os.path.join(labels_folder, labels_path))
        if np.min(y_np) == 0:
            y_np[y_np==0] = 1
        y = torch.tensor(y_np, dtype=torch.long)
        data = Data(x=pos, y=y-1, pos=pos, edge_index=None, edge_attr=None)
        # 3. compute edges (T.Knn)
        edge_creator = T.KNNGraph(k=k)
        data = edge_creator(data)
        dataset.append(data)
    print('\nDataset loaded!')

    return dataset

def dataset_from_pcl(folder, dataset_max_size=50, k=5, data_max_size=90000):

    pcls = os.listdir(folder)
    pcls.sort()
    pcls = pcls[:data_max_size]
    dataset = []
    for pcl_path in pcls:
        pcl = o3d.io.read_point_cloud(os.path.join(folder, pcl_path))
        pos_np = np.asarray(pcl.points)
        pos = torch.tensor(pos_np, dtype=torch.float32)
        y_np = np.asarray(pcl.colors)
        y = torch.tensor(y_np*15, dtype=torch.long)
        data = Data(x=pos[:data_max_size], y=y[:data_max_size, 0], pos=pos[:data_max_size], edge_index=None, edge_attr=None)
        # 3. compute edges (T.Knn)
        breakpoint()
        edge_creator = T.KNNGraph(k=k)
        data = edge_creator(data)
        dataset.append(data)
    return dataset

def prepare_dataset_detection(root_folder, dataset_max_size=50, k=5, data_max_size=90000, \
    use_color=True):

    points_folder = os.path.join(root_folder, 'points')
    colors_folder = os.path.join(root_folder, 'points_color')
    labels_folder = os.path.join(root_folder, 'points_label')
    dataset = []

    scenes = os.listdir(points_folder)
    scenes.sort()
    scenes = scenes[:dataset_max_size]

    for scene in scenes:
        print(f"loading {scene}", end='\r')
        scene_name = scene[:-4]
        pts = torch.tensor(np.loadtxt(os.path.join(points_folder, scene)), dtype=torch.float32)
        if use_color == True:
            cols = torch.tensor(np.loadtxt(os.path.join(colors_folder, f"{scene_name}.col")), dtype=torch.float32)
        labels = torch.tensor(np.loadtxt(os.path.join(labels_folder, f"{scene_name}.seg")), dtype=torch.long)
        # print("max:", np.max(labels.numpy()))
        # print("min:", np.min(labels.numpy()))
        # breakpoint()
        if use_color == True:
            feat_size = 6
        else:
            feat_size = 3
        x = torch.zeros(pts.shape[0], feat_size)
        x[:,:3] = pts
        if use_color == True:
            x[:,3:6] = cols
        data = Data(x=x[:data_max_size], y=labels[:data_max_size], pos=pts[:data_max_size], edge_index=None, edge_attr=None)
        # 3. compute edges (T.Knn)
        edge_creator = T.KNNGraph(k=k)
        data = edge_creator(data)
        dataset.append(data)
        #print(data)
        #breakpoint()
    return dataset