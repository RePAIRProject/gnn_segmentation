import os, sys
import yaml 
import open3d as o3d 
import torch 
import pickle 
from torch_geometric.loader import DataLoader
from utils.train_test_util import predict, training_loop_one_epoch, test_with_loader, \
    show_results, add_noise, print_parameters, show_data, get_loss, get_weights, build_model
from network.gnns import detectionGCN, GAT
import pandas as pd 
import numpy as np 

if __name__ == '__main__':
    
    task = 'detection' # 'recognition' or 'detection'
    print("#" * 50)
    print(f"\nEvaluating fragment {task}\n")
    cfg_file_path = os.path.join(sys.argv[1])
    with open(cfg_file_path, 'r') as yf:
        cfg = yaml.safe_load(yf)
    
    print_parameters(cfg)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} to evaluate..")
    print("#" * 50)
    print('reading data..')
    dataset_name = cfg['dataset_root'].split('/')[-1]
    dataset_path = os.path.join('data', f'dataset_from_{dataset_name}_for_{task}_xyzrgb')
    print('using training data in', dataset_path)
    split_num = cfg['split']
    with open(os.path.join(dataset_path, f'test_set_split_{split_num}'), 'rb') as test_set_file: 
        test_set = pickle.load(test_set_file)

    model = build_model(cfg)
    model.to(device)
    model.load_state_dict(torch.load(cfg['last_model_path']))
    test_loader = DataLoader(test_set, shuffle=True)

    model.eval() 

    correct = 0
    for data in test_loader:
        if cfg['add_noise'] == True:
            data = add_noise(data, cfg['noise_strength'])
        data.to(device)
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)                            # Use the class with highest probability.
        label_class = data.y.argmax(dim=1)
        correct += ((pred == label_class).sum() / out.shape[0]).item()

    acc = (correct / len(test_loader.dataset))

    print("\nFinal Results")
    print(f"Accuracy: {acc:.4f} || {acc*100:.2f}%")

    idx_to_show = np.linspace(0, len(test_set)-1, cfg['how_many']).astype(int)
    for j in idx_to_show:
        data = test_set[j]
        if cfg['add_noise'] == True:
            data = add_noise(data, cfg['noise_strength'])
        pred = predict(model, data, device) # pred returned is already .cpy().numpy()
        pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(data.pos.cpu().numpy()))
        print('pred')
        show_results(pred, pcl, window_name=f"Prediction Scene {j}")
        print('gt')
        # breakpoint()
        labels = (data.y.argmax(dim=1)).cpu().numpy()
        show_results(labels, pcl, window_name=f"Ground Truth Scene {j}")
        
    breakpoint()

    accs = []
    accs_frags = []
    accs_sand = []

    for j, data_sample in enumerate(test_set):
        print(f"predicting and evaluating scene {j:04d}..", end="\r")
        pred = predict(model, data_sample, device) # pred returned is already .cpy().numpy()
        labels = (data_sample.y).cpu().numpy()
        num_pts = labels.shape[0]
        # breakpoint()
        accs.append((num_pts - np.sum(pred!=labels)) / labels.shape[0])
        accs_frags.append((num_pts - np.sum((pred!=labels)*(labels>0))) / labels.shape[0])
        accs_sand.append((num_pts - np.sum((pred!=labels)*(labels==0))) / labels.shape[0])
    
    print("\nFinal Results")
    print("Acc:", np.mean(np.asarray(accs)))
    print("Acc on Frags:", np.mean(np.asarray(accs_frags)))
    print("Acc on Sand:", np.mean(np.asarray(accs)))

    acc_df = pd.DataFrame()
    acc_df['Accuracy'] = accs
    acc_df['Accuracy on Fragments'] = accs_frags
    acc_df['Accuracy on Sand Background'] = accs_sand
    acc_df.to_csv(os.path.join("results", f"{cfg['model_name_save']}.csv"))
