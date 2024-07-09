import yaml 
import torch 
from dataset import prepare_dataset_detection, dataset_from_pcl
from train_test_util import show_results, predict
from net import GCN, GAT
import os, sys 
import open3d as o3d 

if __name__ == '__main__':

    path = sys.argv[1]
    yaml_path = f"checkpoints/{sys.argv[1]}_config.yaml"
    model_path = f"checkpoints/{sys.argv[1]}.pth"
    print(f"looking for:\ncfg in {yaml_path}\nweights in {model_path}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(yaml_path, 'r') as yf:
        cfg = yaml.safe_load(yf)

    input_features = cfg['input_features']
    hidden_channels = cfg['hidden_channels']
    output_classes = cfg['num_seg_classes']
    model_name = cfg['model']
    if model_name == 'GAT':
        model = GAT(input_features=input_features,
                    hidden_channels=hidden_channels,
                    output_classes=output_classes)
    elif model_name == 'GCN':
        model = GCN(input_features=input_features,
                    hidden_channels=hidden_channels,
                    output_classes=output_classes)

    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()

    dataset = prepare_dataset_detection(root_folder = cfg['dataset_root'], \
         dataset_max_size=3, k=cfg['k'], use_color=cfg['use_color'])

    for j in range(3):
        pred = predict(model, dataset[j].to(device), device) # pred returned is already .cpy().numpy()
        pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(dataset[j].pos.cpu().numpy()))
        show_results(pred, pcl)