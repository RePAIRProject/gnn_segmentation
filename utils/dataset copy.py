import torch 
import numpy as np
import torch_geometric.transforms as T
import torch.nn.functional as F
import os 
from torch_geometric.data import Data
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm

def pcl_to_tensor(pcl, k):
    pts = np.asarray(pcl.points)
    col = np.asarray(pcl.colors)
    x = torch.tensor(np.concatenate((pts, col), axis=1), dtype=torch.float32)
    data = Data(x=x, pos=torch.tensor(pts, dtype=torch.float32))
    edge_creator = T.KNNGraph(k=k)
    data = edge_creator(data)
    return data

def dataset_v3(parameters):

    data_max_size = parameters['dataset_max_size']
    points_folder = os.path.join(parameters['dataset_root'], 'points_as_txt')
    labels_folder = os.path.join(parameters['dataset_root'], 'labels')
    dataset = []
    scenes = os.listdir(points_folder)
    scenes.sort()
    scenes = scenes[:data_max_size]
    for scene in scenes:
        print(f"loading {scene}", end='\r')
        scene_name = scene[:-4]
        np_pts = np.loadtxt(os.path.join(points_folder, scene))
        pts = torch.tensor(np_pts, dtype=torch.float32)
        # breakpoint()
        # pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np_pts[:,:3]))
        # pcl.colors=o3d.utility.Vector3dVector(np_pts[:,3:] / 256)
        np_labels = np.loadtxt(os.path.join(labels_folder, f"{scene_name}_labels.txt"))
        labels = torch.tensor(np_labels, dtype=torch.long)
        pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np_pts[:,:3]))
        # breakpoint()
        labels_as_colors = cm.jet(plt.Normalize(min(np_labels),max(np_labels))(np_labels))
        pcl.colors=o3d.utility.Vector3dVector(labels_as_colors[:,:3])
        o3d.visualization.draw_geometries([pcl])
        breakpoint()
        # 
        # # print("max", labels.max().item())
        # print("min", labels.min().item())
        # breakpoint()
        oh_labels = F.one_hot(labels-1, num_classes=6).type(torch.FloatTensor) 
        x = pts 
        if parameters['normalize_color'] == True:
            x[:,3:6] /= 256
        pos = x[:,:3]
        data = Data(x=x[:], y=oh_labels[:], pos=pos[:], edge_index=None, edge_attr=None)
        # 3. compute edges (T.Knn)
        edge_creator = T.KNNGraph(k=parameters['k'])
        data = edge_creator(data)
        dataset.append(data)

    return dataset

def dataset_v2(root_folder, dataset_max_size=50, k=5, data_max_size=90000, \
    use_color=True, normalize_color=False):
    points_folder = os.path.join(root_folder, 'points')
    labels_folder = os.path.join(root_folder, 'labels')
    dataset = []
    scenes = os.listdir(points_folder)
    scenes.sort()
    scenes = scenes[:dataset_max_size]
    for scene in scenes:
        print(f"loading {scene}", end='\r')
        scene_name = scene[:-4]
        pts = torch.tensor(np.loadtxt(os.path.join(points_folder, scene)), dtype=torch.float32)
        labels = torch.tensor(np.loadtxt(os.path.join(labels_folder, f"{scene_name}_labels.txt")), dtype=torch.long)
        if use_color == True:
            feat_size = 6
            x = pts
            if normalize_color == True:
                x[:,3:] /= 256
        else:
            feat_size = 3
            x = pts[:,:3]
        pos = x[:,:3]
        data = Data(x=x[:data_max_size], y=labels[:data_max_size], pos=pos[:data_max_size], edge_index=None, edge_attr=None)
        # 3. compute edges (T.Knn)
        edge_creator = T.KNNGraph(k=k)
        data = edge_creator(data)
        dataset.append(data)
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
        x[:,:3] = pts[:,:3]
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