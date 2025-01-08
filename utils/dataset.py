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

def dataset_of_fragments(parameters):
    """
    It extracts fragments from each scene and 
    create individual files in separate folders
    used to create a classification database 
    """
    # data_max_size = parameters['dataset_max_size']
    points_folder = os.path.join(parameters['dataset_root'], 'points_as_txt')
    labels_folder = os.path.join(parameters['dataset_root'], 'labels')
    dataset = []
    scenes = os.listdir(points_folder)
    scenes.sort()
    # scenes = scenes[:data_max_size]
    # fragments = {}
    
    labels_count = {}
    for k in range(parameters['num_classes']):
        labels_count[k] = 0
    for scene in scenes:
        print(f"loading {scene}", end='\r')
        scene_name = scene[:-4]
        np_pts = np.loadtxt(os.path.join(points_folder, scene))
        if parameters['add_noise'] == True:
            z_range = np.max(np_pts[:,2]) - np.min(np_pts[:,2])
            np_pts[:,2] += np.random.uniform(-1,1,np_pts.shape[0]) * parameters['noise_strength'] * z_range
        np_labels = np.loadtxt(os.path.join(labels_folder, f"{scene_name}_labels.txt"))
        labels = torch.tensor(np_labels, dtype=torch.long)
        if parameters['use_normals'] == True:
            pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np_pts[:,:3]))
            if parameters['use_normals'] == True:
                if pcl.has_normals() == False:
                    pcl.estimate_normals()
                normals = np.asarray(pcl.normals)                     
                np_pts = np.concatenate([np_pts, normals], axis=1)

        # get the fragments
        for val in np.unique(np_labels):
            if val > 1:
                frag = np_pts[np_labels==val]
                # pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(frag[:,:3]))
                # pcl.paint_uniform_color((0,0,1))
                # o3d.visualization.draw_geometries([pcl], window_name=f'Class {val-1}')
                # fragments[f'{int(val):d}'].append(frag)
                pts = torch.tensor(frag, dtype=torch.float32)
                # print(val, parameters['num_frags']+1)
                # breakpoint()
                label_val = int(val - 2)
                y = F.one_hot(torch.tensor([label_val], dtype=torch.long), num_classes=parameters['num_classes']).type(torch.FloatTensor)
                data = Data(x=pts, y=y, pos=pts[:, :3], edge_index=None, edge_attr=None)
                # 3. compute edges (T.Knn)
                edge_creator = T.KNNGraph(k=parameters['k'])
                data = edge_creator(data)
                dataset.append(data)
                labels_count[label_val] += 1
    
    print("\nSTATS")
    for lck in labels_count.keys():
        print("label", lck, ":", labels_count[lck], "objects")
    return dataset

def visualize_points(pos, edge_index=None, index=None):
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
             src = pos[src].tolist()
             dst = pos[dst].tolist()
             plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    else:
       mask = torch.zeros(pos.size(0), dtype=torch.bool)
       mask[index] = True
       plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
       plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    plt.axis('off')
    plt.show()

def dataset_sand_scene(parameters):
    """
    It prepares the dataset for recognition 
    framed as multi-class labeling problem
    (0 for sand, id for the fragments (1,2,3,..) ) 
    """
    # data_max_size = parameters['dataset_max_size']
    points_folder = os.path.join(parameters['dataset_root'], 'points_as_txt')
    labels_folder = os.path.join(parameters['dataset_root'], 'labels')
    dataset = []
    scenes = os.listdir(points_folder)
    scenes.sort()
    # scenes = scenes[:data_max_size]
    for scene in scenes:
        print(f"loading {scene}", end='\r')
        scene_name = scene[:-4]
        np_pts = np.loadtxt(os.path.join(points_folder, scene))
        if parameters['add_preprocessing_noise'] == True:
            z_range = np.max(np_pts[:,2]) - np.min(np_pts[:,2])
            np_pts[:,2] += np.random.uniform(-1,1,np_pts.shape[0]) * parameters['preprocessing_noise_strength'] * z_range
        # breakpoint()
        # pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np_pts[:,:3]))
        # pcl.colors=o3d.utility.Vector3dVector(np_pts[:,3:] / 256)
        np_labels = np.loadtxt(os.path.join(labels_folder, f"{scene_name}_labels.txt"))
        if parameters['task'] == 'detection':
            np_labels = (np_labels > 1).astype(int)
        labels = torch.tensor(np_labels, dtype=torch.long)
        if parameters['use_normals'] == True:
            pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np_pts[:,:3]))
            if parameters['use_normals'] == True:
                if pcl.has_normals() == False:
                    pcl.estimate_normals()
                normals = np.asarray(pcl.normals)                     
                np_pts = np.concatenate([np_pts, normals], axis=1)
            
        # breakpoint()
        # labels_as_colors = cm.jet(plt.Normalize(min(np_labels),max(np_labels))(np_labels))
        # pcl.colors=o3d.utility.Vector3dVector(labels_as_colors[:,:3])
        # o3d.visualization.draw_geometries([pcl])
        # breakpoint()
        # 
        # # print("max", labels.max().item())
        # print("min", labels.min().item())
        # breakpoint()
        if parameters['task'] == 'recognition':
            oh_labels = F.one_hot(labels-1, num_classes=6).type(torch.FloatTensor) 
        elif parameters['task'] == 'detection':
            oh_labels = F.one_hot(labels, num_classes=2).type(torch.FloatTensor) 
         
        # transform into a torch tensor
        pts = torch.tensor(np_pts, dtype=torch.float32)
        x = pts
        if parameters['normalize_color'] == True:
            x[:,3:6] /= 256
        pos = x[:,:3]
        
        # if parameters['task'] == 'detection':
        #     y = labels
        # elif parameters['task'] == 'recognition':
        #     y = oh_labels

        data = Data(x=x[:], y=oh_labels, pos=pos[:], edge_index=None, edge_attr=None)
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
        normals = np.asarray(pcl.normals)
        x[:, 0:3] = pos_np
        x[:, 3:6] = normals
        xt = torch.tensor(x, dtype=torch.float32)
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