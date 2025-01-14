import random
import pickle
from utils.train_test_util import predict, show_results
from torch_geometric.loader import DataLoader
import open3d as o3d
import numpy as np 
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import yaml

def torch_data_from_o3d_pcl(pcl, k, task):
    xyz = np.asarray(pcl.points)
    z_range = np.max(xyz, axis=1) - np.min(xyz, axis=1)
    # xyz[:, 2] += np.random.uniform(-1, 1, xyz.shape[0]) * z_range * 0.1
    rgb = np.asarray(pcl.colors)
    if task == 'detection':
        # colors are okay
        rgb = rgb
    elif task == 'recognition':
        rgb *= 256
    # if normalize_color == True:
    #     rgb = rgb
    # breakpoint()
    pts = torch.from_numpy(np.concatenate([xyz, rgb], axis=1)).type(torch.float32)
    data = Data(x=pts, pos=pts[:, :3], edge_index=None, edge_attr=None)
    # 3. compute edges (T.Knn)
    edge_creator = T.KNNGraph(k=k)
    data = edge_creator(data)
    return data

def recognize_objects(pcls, model, parameters, device):
    labels = np.array(pcls.cluster_dbscan(eps=0.0002, min_points=10, print_progress=False))
    ids = np.linspace(0, labels.shape[0]-1, labels.shape[0]).astype(int)
    datas = []
    recognized_pcls = []
    unique_labels = np.unique(labels)
    print(f"found {len(unique_labels)} objects")
    for label in unique_labels:
        idx_label = [idx for idx in ids if labels[idx] == label]
        pcl = pcls.select_by_index(idx_label)
        recognized_pcls.append(pcl)
        datas.append(torch_data_from_o3d_pcl(pcl, parameters['k'], task='recognition'))        
    dataloader = DataLoader(datas, batch_size=1, shuffle=False)
    # breakpoint()
    pred_IDs = []
    for data in dataloader:
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)  
        pred_class = out.argmax(dim=1)
        pred_IDs.append(pred_class.item())
        #print("predicted", pred_class.item())
            
    return recognized_pcls, pred_IDs


def detect(pcl, detection_model, parameters, device, normalize_color=False, return_tensors=True, debug=False):
    
    data = torch_data_from_o3d_pcl(pcl, parameters['k'], task='detection')
    data.to(device)
    out = detection_model(data.x, data.edge_index)
    pred_class = out.argmax(dim=1).cpu().numpy()
    # show_results(pred_class, pcl)
    # pred_2 = predict(detection_model, data, device)
    
    ids = np.linspace(0, out.shape[0]-1, out.shape[0]).astype(int)
    pcl_idx = [idx for idx in ids if pred_class[idx] > 0]
    objects_pcd = pcl.select_by_index(pcl_idx)
    background_pcd = pcl.select_by_index(pcl_idx, invert=True)
    if return_tensors==False:
        return objects_pcd, background_pcd
    else:
        return objects_pcd, background_pcd, pred_class