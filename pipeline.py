from network.gnns import detectionGCN, recognitionGCN
from utils.train_test_util import predict, show_results
import os 
import random
import pickle
from torch_geometric.loader import DataLoader
import open3d as o3d
import numpy as np 
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data

def detect(pcl, detection_model, parameters, device, normalize_color=False, return_tensors=True, debug=False):
    xyz = np.asarray(pcl.points)
    rgb = np.asarray(pcl.colors)
    if normalize_color == True:
        rgb = rgb * 256 * 256
    pts = torch.from_numpy(np.concatenate([xyz, rgb], axis=1)).type(torch.float32)
    data = Data(x=pts, pos=pts[:, :3], edge_index=None, edge_attr=None)
    # 3. compute edges (T.Knn)
    edge_creator = T.KNNGraph(k=parameters['k'])
    data = edge_creator(data)
    data.to(device)
    out = detection_model(data.x, data.edge_index)
    pred_class = out.argmax(dim=1).cpu().numpy()
    show_results(pred_class, pcl)
    # pred_2 = predict(detection_model, data, device)
    
    ids = np.linspace(0, out.shape[0]-1, out.shape[0]).astype(int)
    pcl_idx = [idx for idx in ids if pred_class[idx] > 0]
    objects_pcd = pcl.select_by_index(pcl_idx)
    background_pcd = pcl.select_by_index(pcl_idx, invert=True)
    if return_tensors==False:
        return objects_pcd, background_pcd
    else:
        return objects_pcd, background_pcd, pred_class

if __name__ == '__main__':



    # dataset_root_path = '/media/lucap/big_data/datasets/repair/sand_detection_dataset_bb_yolo_1000scenes'
    # pcl = o3d.io.read_point_cloud(os.path.join(dataset_root_path, 'group_0015', 'pointclouds', f'PCL_000000.ply'))
    # xyz = np.asarray(pcl.points)
    # rgb = np.asarray(pcl.colors)    
    # data_pts = test_set[0].x
    # np_pts = np.concatenate([xyz, rgb], axis=1)
    # breakpoint()
    # pts = torch.from_numpy(np_pts).type(torch.float32).to(device)
    

    # MODELS
    detection_model_folder = 'checkpoints/sand_scene-detection-net_GCN-based_trained_on_sand_detection_dataset_bb_yolo_1000scenes_split_5_using_lossCAT_for10epochs_from_scratch_bs_1_noiseTrue'
    detection_model_path = os.path.join(detection_model_folder, 'model.pt')
    detection_best_weights = os.path.join(detection_model_folder, 'last.pth')
    parameters = {'k': 10}
    
    g15_recognition_model_folder = 'checkpoints/fragment-recognition-net_GCN-based_trained_on_sand_detection_dataset_bb_yolo_1000scenes_split_7-group_0015_using_lossCAT_for500epochs_from_scratch_bs_128_noiseTrue'
    g15_recognition_model_path = os.path.join(g15_recognition_model_folder, 'model.pt')
    best_weights_g15 = os.path.join(g15_recognition_model_folder, 'best.pth')
    g29_recognition_model_path = 'checkpoints/fragment-recognition-net_GCN-based_trained_on_sand_detection_dataset_bb_yolo_1000scenes_split_7-group_0029_using_lossCAT_for500epochs_from_scratch_bs_128_noiseTrue/model.pt'

    # DATASET
    dataset_root_path = '/media/lucap/big_data/datasets/repair/sand_detection_dataset_bb_yolo_1000scenes'
    
    # CUDA DEVICE 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # READ A RANDOM POINTCLOUD
    random_scene_idx = np.random.choice(1000)
    pcl_15 = o3d.io.read_point_cloud(os.path.join(dataset_root_path, 'group_0015', 'pointclouds', f'PCL_{random_scene_idx:06d}.ply'))
    o3d.visualization.draw_geometries([pcl_15], window_name='Input Pointcloud')

    detection_model = torch.load(detection_model_path).to(device)
    detection_model.load_state_dict(torch.load(detection_best_weights))
    detection_model.eval() 
    objects, background, preds = detect(pcl_15, detection_model, parameters, device)

    objects.paint_uniform_color([0,0,1])
    background.paint_uniform_color([1,0,0])
    o3d.visualization.draw_geometries([objects, background])
    

    pcl_29 = o3d.io.read_point_cloud(os.path.join(dataset_root_path, 'group_0029', 'pointclouds', f'PCL_{random_scene_idx:06d}.ply'))