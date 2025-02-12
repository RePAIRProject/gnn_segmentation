import torch 
from utils.dataset import prepare_dataset_detection, dataset_from_pcl, dataset_v2, dataset_v3
from network.net import GCN, GAT, recognitionGCN
from torch_geometric.loader import DataLoader
from utils.train_test_util import predict, training_loop_one_epoch, test_with_loader, \
    show_results
import os, json
import open3d as o3d 
import numpy as np 
import yaml 
import shutil 
import pickle 

if __name__ == '__main__':

    task = 'recognition' # 'recognition' or 'detection'
    group = 15
    num_frags = 17
    num_classes = num_frags+1
    print("#" * 50)
    print(f"\nTraining for {task}\n")
    cfg_file_path = os.path.join('configs', f'cfg_{task[:3]}.yaml')
    with open(cfg_file_path, 'r') as yf:
        cfg = yaml.safe_load(yf)
    
    # adjust for this group
    cfg['num_frags'] = num_frags
    cfg['num_seg_classes'] = num_classes
    cfg['dataset_root'] = os.path.join(cfg['dataset_root'], f'group_{group:04d}')
    print("#" * 50)
    print("# PARAMETERS")
    print("#" * 50)
    for cfg_key in cfg.keys():
        print(f"# {cfg_key}:{cfg[cfg_key]}")
    print("#" * 50)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} to train..")
    print('reading data..')
    with open(os.path.join('data', f'group_{group}_fragments_{task}_training_set'), 'rb') as training_set_file: 
        train_dataset = pickle.load(training_set_file)
    with open(os.path.join('data', f'group_{group}_fragments_{task}_test_set'), 'rb') as test_set_file: 
        test_dataset = pickle.load(test_set_file)

    print('model..')
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

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=5e-4)
    if cfg['task'] == 'detection':
        weight = torch.tensor([1, cfg['weight_obj']], dtype=torch.float32).to(device)
    elif cfg['task'] == 'recognition':
        weights = np.ones((cfg['num_frags']+1)) * cfg['weight_obj']
        weights[0] /= 2
        weight = torch.tensor(weights, dtype=torch.float32).to(device)
    if cfg['loss'] == "NLL":
        criterion = torch.nn.NLLLoss(weight=weight) #()
    # elif cfg['loss'] == "CAT":
    #     criterion = torch.nn.CategoricalCrossEntropyLoss(weight=weight)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=weight) #NLLLoss()

    print("start training..")
    EPOCHS = cfg['epochs']
    test_acc = 0.0
    acc_intact = 0.0
    acc_broken = 0.0
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=True)

    if cfg['continue_training'] == True:
        cnt = "continuation"
        model.load_state_dict(torch.load(cfg['ckp_path'], weights_only=True))
    else:
        cnt = 'from_scratch'
    model.train()
    best_loss = 1
    base_name = cfg['dataset_root'].split('/')[-1]
    
    model_name_save = f"fragment-{task}-net_{model_name}-based_trained_on_{base_name}_using_loss{cfg['loss']}_for{EPOCHS}epochs_{cnt}_bs_{cfg['batch_size']}"
    best_model_name = ""
    for epoch in range(0, EPOCHS):
        # loss = training_loop_one_epoch(model, train_loader, criterion, optimizer, device)
        for data in train_loader:  # Iterate in batches over the training dataset.
            data.to(device)
            out = model(data.x, data.edge_index, data.batch)    # Perform a single forward pass.
            loss = criterion(out, data.y)           # Compute the loss.
            loss.backward()                         # Derive gradients.
            optimizer.step()                        # Update parameters based on gradients.
            optimizer.zero_grad()                   # Clear gradients.

        # print(loss.item())

        if (epoch+1) % cfg['print_each'] == 0:
            #pdb.set_trace()
            print(f'Epoch: {(epoch+1):03d}, Loss: {loss:.4f}')
            
        if loss.item() < best_loss:
            best_model_name = f"{model_name_save}_BEST_after_{epoch+1}_epochs"
            torch.save(model.state_dict(), os.path.join(cfg['models_path'], best_model_name))
    
    torch.save(model.state_dict(), os.path.join(cfg['models_path'], model_name_save))
    
    cfg['base_name'] = base_name
    cfg['model_name_save'] = model_name_save
    cfg['best_model_name'] = best_model_name
    cfg['last_model_path'] = os.path.join(cfg['models_path'], model_name_save)
    cfg['best_model_path'] = os.path.join(cfg['models_path'], best_model_name)

    res_cfg_path = os.path.join(cfg['models_path'], f"{model_name_save}_config.yaml")
    with open(res_cfg_path, 'w') as yf:
        yaml.dump(cfg, yf)
        
    # shutil.copy(cfg_file_path, os.path.join(cfg['models_path'], f"{model_name_save}_config.yaml"))
    print(f"saved {model_name_save}")
    print(f"For inference, run:")
    print(f"\npython evaluate_fragment_recognition.py {res_cfg_path}\n")
    
    if cfg['show_results'] == True:
        print(f"showing {cfg['how_many']} results..")
        model.eval()
        
        # idx_to_show = np.linspace(0, len(test_dataset)-1, cfg['how_many']).astype(int)
        counter = 0
        for data in test_loader:
            if counter > cfg['how_many']:
                continue
            out = model(data.x, data.edge_index, data.batch)  
            pred_class = out.argmax(dim=1)
            label_class = data.y.argmax(dim=1)
            print('-' * 40)
            print(f"Prediction for scene {j}:")
            print(pred, '\nclass:', pred_class)
            print("correct class:", label_class)
            # pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(test_dataset[j].pos.cpu().numpy()))
            # print('pred')
            # show_results(pred, pcl, window_name=f"Prediction Scene {j}")
            # print('gt')
            # # breakpoint()
            # labels = (test_dataset[j].y).cpu().numpy()
            # show_results(labels, pcl, window_name=f"Ground Truth Scene {j}")
            # breakpoint()
