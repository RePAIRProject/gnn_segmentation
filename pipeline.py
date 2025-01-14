from network.gnns import detectionGCN, recognitionGCN
from utils.train_test_util import predict, show_results, add_noise
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
from utils.integration import detect, torch_data_from_o3d_pcl, recognize_objects


if __name__ == '__main__':

    # dataset_root_path = '/media/lucap/big_data/datasets/repair/sand_detection_dataset_bb_yolo_1000scenes'
    # pcl = o3d.io.read_point_cloud(os.path.join(dataset_root_path, 'group_0015', 'pointclouds', f'PCL_000000.ply'))
    # xyz = np.asarray(pcl.points)
    # rgb = np.asarray(pcl.colors)    
    # data_pts = test_set[0].x
    # np_pts = np.concatenate([xyz, rgb], axis=1)
    # breakpoint()
    # pts = torch.from_numpy(np_pts).type(torch.float32).to(device)
    print('reading config file..')
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
    g15_recognition_model_weights = os.path.join(g15_recognition_model_folder, 'best.pth')
    best_weights_g15 = os.path.join(g15_recognition_model_folder, 'best.pth')
    g29_recognition_model_path = cfg['g29_recognition_model_path']
    #'checkpoints/fragment-recognition-net_GCN-based_trained_on_sand_detection_dataset_bb_yolo_1000scenes_split_7-group_0029_using_lossCAT_for500epochs_from_scratch_bs_128_noiseTrue/model.pt'

    # DATASET
    dataset_root_path = cfg['dataset_root_path']
    #'/media/lucap/big_data/datasets/repair/sand_detection_dataset_bb_yolo_1000scenes'
    
    # CUDA DEVICE 
    print('device')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # LOAD THE MODEL 
    print("detection model")
    detection_model = detectionGCN(6,64)
    weights_dict = torch.load(detection_best_weights, map_location=device)
    detection_model.load_state_dict(state_dict=weights_dict)
    # if device == torch.device('cpu'):
    #     detection_model2 = torch.load(detection_model_path, map_location=torch.device('cpu'))
    # else:
    #     detection_model = torch.load(detection_model_path).to(device)
    # detection_model = torch.load(detection_model_path).to(device)
    # detection_model.load_state_dict(torch.load(detection_best_weights))
    detection_model.eval() 

    # LOAD DATA
    # with open('/media/palma/D36D-688D/uni palma/gnn_segmentation/data/detection_sand_test_set', 'rb') as valid_set_file: 
    #     validation_set = pickle.load(valid_set_file)
    # valid_loader = DataLoader(validation_set, batch_size=1, shuffle=True)
    # for data in valid_loader:  # Iterate in batches over the training dataset.
    #     # ADD NOISE
    #     data = add_noise(data, 0.5)
    #     data.x = data.x[:,:6]
    #     data.to(device)
    #     out = detection_model(data.x, data.edge_index)
    #     print("unique values", out.argmax(dim=1).unique())
    # breakpoint()

    # READ A RANDOM POINTCLOUD
    print("read random pointcloud..")
    random_scene_idx = np.random.choice(100)
    pcl_15 = o3d.io.read_point_cloud(os.path.join(dataset_root_path, 'group_0015', 'pointclouds', f'PCL_{random_scene_idx:06d}.ply'))
    o3d.visualization.draw_geometries([pcl_15], window_name='Input Pointcloud')

    # DETECT
    print('detecting..')
    detected_objects, background, preds = detect(pcl_15, detection_model, parameters, device)
    detected_objects.paint_uniform_color([0,0,1])
    background.paint_uniform_color([1,0,0])
    o3d.visualization.draw_geometries([detected_objects, background], window_name = 'detected objects (blue) and background (red)')

    print('recognition model')
    recognition_model_g15 = recognitionGCN(6, 64, 17, 0.5)
    g15_weights_dict = torch.load(g15_recognition_model_weights, map_location=device)
    recognition_model_g15.load_state_dict(state_dict=g15_weights_dict)
    recognition_model_g15.eval()
    # recognition_model_g29 = recognitionGCN(6, 64, 17, 0.5)
    # weights_dict = torch.load(g29_recognition_model_weights, map_location=device)
    # recognition_model_g29.eval()
    print('recognizing..')
    recognized_objects_as_pcls, pred_IDs = recognize_objects(detected_objects, recognition_model_g15, parameters, device)
    # max_label = np.max(pred_IDs)
    # colors = plt.get_cmap("tab20")(pred_IDs / (max_label if max_label > 0 else 1))
    # recognized_objects_as_pcls.colors = o3d.utility.Vector3dVector(colors[:, :3])
    pcl_to_show = []
    for rop, id in zip(recognized_objects_as_pcls, pred_IDs):
        print('id', id)
        if id == 0:
            rop.paint_uniform_color([0, 0, 1])
            pcl_to_show.append(rop)
        elif id == 1:
            rop.paint_uniform_color([0, 1, 0])
            pcl_to_show.append(rop)
        elif id == 2:
            rop.paint_uniform_color([1, 0, 0])
            pcl_to_show.append(rop)
    o3d.visualization.draw_geometries(pcl_to_show)
    breakpoint()

    pcl_29 = o3d.io.read_point_cloud(os.path.join(dataset_root_path, 'group_0029', 'pointclouds', f'PCL_{random_scene_idx:06d}.ply'))