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
from torch_geometric.loader import DataLoader
import yaml

def recognize_objects(pcls):
    datas = []
    for pcl in pcls:
        
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=True)


def detect(pcl, detection_model, parameters, device, normalize_color=False, return_tensors=True, debug=False):
    xyz = np.asarray(pcl.points)
    xyz[:, 2] += np.random.uniform(-1, 1, xyz.shape[0])
    rgb = np.asarray(pcl.colors)
    if normalize_color == True:
        rgb = rgb
    pts = torch.from_numpy(np.concatenate([xyz, rgb], axis=1)).type(torch.float32)
    data = Data(x=pts, pos=pts[:, :3], edge_index=None, edge_attr=None)
    # 3. compute edges (T.Knn)
    edge_creator = T.KNNGraph(k=parameters['k_det'])
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
    cfg_file_path = os.path.join('configs', f'pipeline_cfg.yaml')
    with open(cfg_file_path, 'r') as yf:
        cfg = yaml.safe_load(yf)

    # MODELS
    detection_model_folder = cfg['detection_model_folder']
    detection_model_path = os.path.join(detection_model_folder, 'model.pt')
    detection_best_weights = os.path.join(detection_model_folder, 'last.pth')
    parameters = {'k_det': 10, 'k_rec': 6}
    
    g15_recognition_model_folder = cfg['g15_recognition_model_folder']
    #'checkpoints/fragment-recognition-net_GCN-based_trained_on_sand_detection_dataset_bb_yolo_1000scenes_split_7-group_0015_using_lossCAT_for500epochs_from_scratch_bs_128_noiseTrue'
    g15_recognition_model_path = os.path.join(g15_recognition_model_folder, 'model.pt')
    best_weights_g15 = os.path.join(g15_recognition_model_folder, 'best.pth')
    g29_recognition_model_path = cfg['g29_recognition_model_path']
    #'checkpoints/fragment-recognition-net_GCN-based_trained_on_sand_detection_dataset_bb_yolo_1000scenes_split_7-group_0029_using_lossCAT_for500epochs_from_scratch_bs_128_noiseTrue/model.pt'

    # DATASET
    dataset_root_path = cfg['dataset_root_path']
    #'/media/lucap/big_data/datasets/repair/sand_detection_dataset_bb_yolo_1000scenes'
    
    # CUDA DEVICE 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # READ A RANDOM POINTCLOUD
    random_scene_idx = np.random.choice(100)
    pcl_15 = o3d.io.read_point_cloud(os.path.join(dataset_root_path, 'group_0015', 'pointclouds', f'PCL_{random_scene_idx:06d}.ply'))
    o3d.visualization.draw_geometries([pcl_15], window_name='Input Pointcloud')

    
    detection_model = detectionGCN(6,64)
    weights_dict = torch.load(f"{detection_model_path[:-8]}best.pth", map_location=torch.device('cpu'))
    # detection_model.load_state_dict(state_dict=weights_dict)
    # if device == torch.device('cpu'):
    #     detection_model2 = torch.load(detection_model_path, map_location=torch.device('cpu'))
    # else:
    #     detection_model = torch.load(detection_model_path).to(device)
    # detection_model = torch.load(detection_model_path).to(device)
    # detection_model.load_state_dict(torch.load(detection_best_weights))
    detection_model.eval() 
    detected_objects, background, preds = detect(pcl_15, detection_model, parameters, device)
    breakpoint()

    detected_objects.paint_uniform_color([0,0,1])
    background.paint_uniform_color([1,0,0])
    o3d.visualization.draw_geometries([objects, background])

    recognized_objects = recognize(detected_objects)

    pcl_29 = o3d.io.read_point_cloud(os.path.join(dataset_root_path, 'group_0029', 'pointclouds', f'PCL_{random_scene_idx:06d}.ply'))