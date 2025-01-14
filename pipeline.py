from network.gnns import detectionGCN, recognitionGCN
from utils.train_test_util import predict, show_results, add_noise
import os, sys
import random
import pickle
from torch_geometric.loader import DataLoader
import open3d as o3d
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import yaml
from utils.integration import detect, torch_data_from_o3d_pcl, recognize_objects
import copy

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


    
   
    # DATASET
    dataset_root_path = cfg['dataset_root_path']
    #'/media/lucap/big_data/datasets/repair/sand_detection_dataset_bb_yolo_1000scenes'
    
    # CUDA DEVICE 
    print('device')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # LOAD THE MODEL 
    print("detection model")
    # MODELS
    detection_model_folder = cfg['detection_model_folder']
    # detection_model_path = os.path.join(detection_model_folder, 'model.pt')
    detection_best_weights = os.path.join(detection_model_folder, 'last.pth')
    parameters_det = {'input_features': 6, 'hidden_channels': 64, 'k': 10}
    detection_model = detectionGCN(input_features=parameters_det['input_features'], \
        hidden_channels=parameters_det['hidden_channels'])
    weights_dict = torch.load(detection_best_weights, map_location=device)
    detection_model.load_state_dict(state_dict=weights_dict)
    detection_model.eval() 

    # if device == torch.device('cpu'):
    #     detection_model2 = torch.load(detection_model_path, map_location=torch.device('cpu'))
    # else:
    #     detection_model = torch.load(detection_model_path).to(device)
    # detection_model = torch.load(detection_model_path).to(device)
    # detection_model.load_state_dict(torch.load(detection_best_weights))

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
    if len(sys.argv) > 1:
        group = sys.argv[1]
    else:
        group = '15'
    print(f"read random pointcloud of group {group}..")
    random_scene_idx = np.random.choice([995, 996, 997, 998, 999])
    pcl_init = o3d.io.read_point_cloud(os.path.join(dataset_root_path, f'group_00{group}', 'pointclouds', f'PCL_{random_scene_idx:06d}.ply'))
    o3d.visualization.draw_geometries([pcl_init], window_name='Input Pointcloud')

    # DETECT
    print('detecting..')
    detected_objects, background, preds = detect(pcl_init, detection_model, parameters_det, device)
    vis_detected_objects = copy.deepcopy(detected_objects)
    vis_detected_objects.paint_uniform_color([0,0,1])
    vis_background = copy.deepcopy(background)
    vis_background.paint_uniform_color([1,0,0])
    o3d.visualization.draw_geometries([vis_detected_objects, vis_background], window_name = 'detected objects (blue) and background (red)')

    print(f'recognition model for group {group}')
    if group == '15':
        g15_recognition_model_folder = cfg['g15_recognition_model_folder']
        # g15_recognition_model_path = os.path.join(g15_recognition_model_folder, 'model.pt')
        g15_recognition_model_weights = os.path.join(g15_recognition_model_folder, 'last.pth')
        parameters_rec_g15 = {'input_features': 6, 'hidden_channels': 64, 'output_classes': 17, 'k': 6}
        recognition_model_g15 = recognitionGCN(input_features=parameters_rec_g15['input_features'], \
            hidden_channels=parameters_rec_g15['hidden_channels'], output_classes=parameters_rec_g15['output_classes'])
        g15_weights_dict = torch.load(g15_recognition_model_weights, map_location=device)
        recognition_model_g15.load_state_dict(state_dict=g15_weights_dict)
        recognition_model_g15.eval()
        print('recognizing..')
        recognized_objects_as_pcls, pred_IDs = recognize_objects(detected_objects, recognition_model_g15, parameters_rec_g15, device)
    elif group == '29':
        g29_recognition_model_folder = cfg['g29_recognition_model_folder']
        g29_recognition_model_weights = os.path.join(g29_recognition_model_folder, 'last.pth')
        parameters_rec_g29 = {'input_features': 6, 'hidden_channels': 64, 'output_classes': 5, 'k': 6}
        recognition_model_g29 = recognitionGCN(input_features=parameters_rec_g29['input_features'], \
            hidden_channels=parameters_rec_g29['hidden_channels'], output_classes=parameters_rec_g29['output_classes'])
        g29_weights_dict = torch.load(g29_recognition_model_weights, map_location=device)
        recognition_model_g29.load_state_dict(state_dict=g29_weights_dict)
        recognition_model_g29.eval()
        print('recognizing..')
        recognized_objects_as_pcls, pred_IDs = recognize_objects(detected_objects, recognition_model_g29, parameters_rec_g29, device)
        
        
        # max_label = np.max(pred_IDs)
    # colors = plt.get_cmap("tab20")(pred_IDs / (max_label if max_label > 0 else 1))
    # recognized_objects_as_pcls.colors = o3d.utility.Vector3dVector(colors[:, :3])
    pcl_to_show = []
    bboxes = []
    if group == '15':
        cmap = plt.get_cmap('jet').resampled(17)
    elif group == '29':
        cmap = plt.get_cmap('jet').resampled(5)
    for rop, pid in zip(recognized_objects_as_pcls, pred_IDs):
        print('object with id', pid)
        col = cmap(pid)[:3]
        rop.paint_uniform_color(col)
        bbox = rop.get_axis_aligned_bounding_box()
        bbox.color = col
        pcl_to_show.append(rop)
        pcl_to_show.append(bbox)
        bboxes.append(bbox)

    o3d.visualization.draw_geometries(pcl_to_show, window_name = f'recognition on group {group} (objects)')
    o3d.visualization.draw_geometries([pcl_init]+bboxes, window_name = f'recognition on group {group} (scene)')
    breakpoint()

