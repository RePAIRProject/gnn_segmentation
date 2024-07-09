import yaml 
import torch 
from dataset import prepare_dataset_detection, dataset_from_pcl
from net import GCN, GAT
import os, sys 

if __name__ == '__main__':

    path = sys.argv[1]
    breakpoint()
    with open(f"{sys.argv[1]}_config.yaml", 'r') as yf:
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

    model.load_state_dict(torch.load(f"{sys.argv[1]}.pth"))
    model.eval()