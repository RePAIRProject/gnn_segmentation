import os, sys
import yaml 
import open3d as o3d 
import torch 
import pickle 
from utils.train_test_util import predict, show_results
from network.gnns import GCN, GAT, recognitionGCN
from torch_geometric.loader import DataLoader
import pandas as pd 
import numpy as np 

if __name__ == '__main__':
    
    task = 'recognition' # 'recognition' or 'detection'
    print("#" * 50)
    print(f"\nEvaluating for {task}\n")
    cfg_file_path = os.path.join(sys.argv[1])
    with open(cfg_file_path, 'r') as yf:
        cfg = yaml.safe_load(yf)
    
    # adjust for this group
    group = cfg['group']
    dataset_name = cfg['dataset_root'].split('/')[-2]
    cfg['dataset_root'] = os.path.join(cfg['dataset_root'], f'group_{group:04d}')
    cfg['batch_size'] = 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} to evaluate..")
    print("#" * 50)
    print('reading data..')
    dataset_path = os.path.join('data', f'dataset_from_{dataset_name}_group_{group}_fragments_{task}_xyzrgb')
    print('using training data in', dataset_path)
    split_num = 7
    with open(os.path.join(dataset_path, f'test_set_split_{split_num}'), 'rb') as test_set_file: 
        test_set = pickle.load(test_set_file)

    # # show data
    # for k in range(0, 10):
    #     pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(test_dataset[k].x[:,:3]))
    #     pcl.paint_uniform_color((0,0,1))
    #     name = f"object of class {int(test_datsplit_num = 7aset[k].y.argmax(dim=1))}"
    #     o3d.visualization.draw_geometries([pcl], window_name = name)
    # breakpoint()
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=True)

    print("#" * 50)
    print("Model")
    input_features = cfg['input_features']
    hidden_channels = cfg['hidden_channels']
    output_classes = cfg['num_classes']
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
                    output_classes=output_classes,
                    dropout=cfg['dropout_rate'])
    else:
        print("WHICH MODEL?")
    model.to(device)
    # breakpoint()
    model.load_state_dict(torch.load(cfg['last_model_path']))#, weights_only=True))
    
    print('-' * 40)
    print("evaluating..")
    model.eval() 
    correct_per_class = np.zeros(cfg['num_classes'])
    total_per_class = np.zeros(cfg['num_classes'])
    correct = 0
    total = 0
    for data in test_loader:
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)  
        pred_class = out.argmax(dim=1)
        label_class = data.y.argmax(dim=1)
        print(f"predicted: {pred_class.item()} // correct: {label_class.item()} \n pred_values({out.cpu().detach().numpy()})")
        # print("correct class:", label_class)
        # breakpoint()
        total += 1
        total_per_class[label_class-1] += 1
        if pred_class == label_class:
            correct += 1
            correct_per_class[label_class-1] += 1

    print(f"Acc: {(correct/total*100):.02f}%")
    print('-' * 40)
    for lc in range(cfg['num_classes']):
        print(f"Class {lc}: {correct_per_class[lc]} / {total_per_class[lc]} ({(correct_per_class[lc]/total_per_class[lc]):.02f})")
    
    # full_model_path = os.path.join(cfg['models_path'], cfg['model_folder'], 'model.pt')
    # if cfg['save_full_model'] == True and not os.path.exists(full_model_path):
    #     torch.save(model, full_model_path)
    
    # # SCRIPTED
    # scripted_model_path = os.path.join(cfg['models_path'], cfg['model_folder'], 'model_scripted.pt')
    # if cfg['save_scripted_model'] == True and not os.path.exists(scripted_model_path):
    #     model_scripted = torch.jit.script(model)
    #     model_scripted.save(scripted_model_path)
        # load with 
        # model = torch.jit.load('model_scripted.pt')

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
