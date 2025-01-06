import os, sys
import yaml 
import open3d as o3d 
import torch 
import pickle 
from utils.train_test_util import predict, show_results
from network.net import GCN, GAT, recognitionGCN
from torch_geometric.loader import DataLoader
import pandas as pd 
import numpy as np 

if __name__ == '__main__':
    
    task = 'recognition' # 'recognition' or 'detection'
    group = 29
    num_frags = 5
    num_classes = 6
    print("#" * 50)
    print(f"\nTraining for {task}\n")
    cfg_file_path = os.path.join(sys.argv[1])
    with open(cfg_file_path, 'r') as yf:
        cfg = yaml.safe_load(yf)
    
    # adjust for this group
    cfg['num_frags'] = num_frags
    cfg['num_seg_classes'] = num_classes
    cfg['dataset_root'] = os.path.join(cfg['dataset_root'], f'group_{group:04d}')
    cfg['batch_size'] = 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} to evaluate..")
    print("#" * 50)
    print('reading data..')
    with open(os.path.join('data', f'group_{group}_fragments_{task}_test_set'), 'rb') as test_set_file: 
        test_dataset = pickle.load(test_set_file)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=True)

    print("#" * 50)
    print("Model")
    input_features = cfg['input_features']
    hidden_channels = cfg['hidden_channels']
    output_classes = cfg['num_seg_classes']
    model_name = cfg['model']
    print(f"{model_name} Model with: \
          {input_features} input features, \
          {hidden_channels} hidden_channels and \
          {output_classes} output_classes")
    # 4. create GCN model
    if model_name == 'GAT':
        model = GAT(input_features=input_features,
                    hidden_channels=hidden_channels,
                    output_classes=output_classes)
    elif model_name == 'GCN':
        model = recognitionGCN(input_features=input_features,
                    hidden_channels=hidden_channels,
                    output_classes=output_classes)
    else:
        print("WHICH MODEL?")
    model.to(device)
    model.load_state_dict(torch.load(cfg['last_model_path'], weights_only=True))
    
    print('-' * 40)
    print("evaluating..")
    model.eval() 
    correct = 0
    total = 0
    for data in test_loader:
        out = model(data.x, data.edge_index, data.batch)  
        pred_class = out.argmax(dim=1)
        label_class = data.y.argmax(dim=1)
        print(f"prediction: {pred_class} \n({out.detach().numpy()})")
        print("correct class:", label_class)
        total += 1
        if pred_class == label_class:
            correct += 1
    print(f"Acc: {(correct/total*100):.02f}%")

    # for j, data_sample in enumerate(test_dataset):
    #     print(f"predicting and evaluating scene {j:04d}..", end="\r")
    #     pred = predict(model, data_sample, device) # pred returned is already .cpy().numpy()
    #     labels = (data_sample.y).cpu().numpy()
    #     num_pts = labels.shape[0]
    #     # breakpoint()
    #     accs.append((num_pts - np.sum(pred!=labels)) / labels.shape[0])
    #     accs_frags.append((num_pts - np.sum((pred!=labels)*(labels>0))) / labels.shape[0])
    #     accs_sand.append((num_pts - np.sum((pred!=labels)*(labels==0))) / labels.shape[0])
    
    # print("\nFinal Results")
    # print("Acc:", np.mean(np.asarray(accs)))
    # print("Acc on Frags:", np.mean(np.asarray(accs_frags)))
    # print("Acc on Sand:", np.mean(np.asarray(accs)))

    # acc_df = pd.DataFrame()
    # acc_df['Accuracy'] = accs
    # acc_df['Accuracy on Fragments'] = accs_frags
    # acc_df['Accuracy on Sand Background'] = accs_sand
    # acc_df.to_csv(os.path.join("results", f"{cfg['model_name_save']}.csv"))
