import yaml 
import torch 
from dataset import prepare_dataset_detection, \
    dataset_from_pcl, dataset_v2, pcl_to_tensor
from train_test_util import show_results, predict
from net import GCN, GAT
import os, sys 
import open3d as o3d 

if __name__ == '__main__':

    path = sys.argv[1]
    yaml_path = f"{sys.argv[1]}_config.yaml"
    if yaml_path.find("/") == -1:
        yaml_path = os.path.join('checkpoints', yaml_path)
    model_path = f"{sys.argv[1]}.pth"
    if model_path.find("/") == -1:
        model_path = os.path.join('checkpoints', model_path)
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

    if torch.cuda.is_available() == True:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device).eval()

    dataset = dataset_v2(root_folder = cfg['dataset_root'], \
         dataset_max_size=3, k=cfg['k'], use_color=cfg['use_color'], normalize_color=cfg['normalize_color'])

    if len(sys.argv) > 2:
        predict_on_input_file = True
        predict_on_dataset = False
        file_path = sys.argv[2]
        print("Predicting on", file_path)
    else:
        predict_on_dataset = True 
        predict_on_input_file = False
        print("Predicting on the dataset")

    if predict_on_dataset == True:
        for j in range(3):
            pred = predict(model, dataset[j].to(device), device) # pred returned is already .cpy().numpy()
        pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(dataset[j].pos.cpu().numpy()))
        show_results(pred, pcl)
    elif predict_on_input_file == True:
        pcl = o3d.io.read_point_cloud(file_path)
        data = pcl_to_tensor(pcl, k=cfg['k'])
        pred = predict(model, data, device)
        show_results(pred, pcl)
    