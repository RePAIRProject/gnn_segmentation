import torch 
from torch_geometric.loader import DataLoader
from utils.train_test_util import predict, training_loop_one_epoch, test_with_loader, \
    show_results, add_noise, print_parameters, show_data, get_loss, get_weights, build_model
import os, json
import open3d as o3d 
import numpy as np 
import yaml 
import shutil 
import pickle 
import pandas as pd 

if __name__ == '__main__':

    task = 'detection' # 'recognition' or 'detection'
    print("#" * 50)
    print(f"\nTraining for {task}\n")
    cfg_file_path = os.path.join('configs', f'cfg_{task[:3]}.yaml')
    with open(cfg_file_path, 'r') as yf:
        cfg = yaml.safe_load(yf)
    
    print("#" * 50)
    print("# PARAMETERS")
    print("#" * 50)
    for cfg_key in cfg.keys():
        print(f"# {cfg_key}:{cfg[cfg_key]}")
    print("#" * 50)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} to train..")
    print('reading data..')
    dataset_name = cfg['dataset_root'].split('/')[-1]
    dataset_path = os.path.join('data', f'dataset_from_{dataset_name}_for_{task}_xyzrgb')
    print('using training data in', dataset_path)
    split_num = 5
    with open(os.path.join(dataset_path, f'training_set_split_{split_num}'), 'rb') as training_set_file: 
        training_set = pickle.load(training_set_file)
    with open(os.path.join(dataset_path, f'validation_set_split_{split_num}'), 'rb') as valid_set_file: 
        validation_set = pickle.load(valid_set_file)
    # with open(os.path.join(dataset_path, f'test_set_split_{split_num}'), 'rb') as test_set_file: 
    #     test_set = pickle.load(test_set_file)

    DEBUG = False
    if DEBUG == True:
        show_data(validation_set, 10)

    print('model..')   
    model = build_model(cfg)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=5e-4)

    weight = get_weights(cfg)
    weight = weight.to(device)

    criterion = get_loss(cfg, weight)
    # print('model..')
    # input_features = cfg['input_features']
    # hidden_channels = cfg['hidden_channels']
    # output_classes = cfg['num_seg_classes']
    # model_name = cfg['model']
    # print(f"{model_name} Model with: \
    #       {input_features} input features, \
    #       {hidden_channels} hidden_channels and \
    #       {output_classes} output_classes")
    # # 4. create GCN model
    # if model_name == 'GAT':
    #     model = GAT(input_features=input_features,
    #                 hidden_channels=hidden_channels,
    #                 output_classes=output_classes)
    # elif model_name == 'GCN':
    #     model = GCN(input_features=input_features,
    #                 hidden_channels=hidden_channels,
    #                 output_classes=output_classes)
    # else:
    #     print("WHICH MODEL?")

    # model.to(device)

    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=cfg['lr'], weight_decay=5e-4)
    # if cfg['task'] == 'detection':
    #     weight = torch.tensor([1, cfg['weight_obj']], dtype=torch.float32).to(device)
    # elif cfg['task'] == 'recognition':
    #     weight = torch.tensor([cfg['weight_obj']/2, cfg['weight_obj'], cfg['weight_obj'], cfg['weight_obj'], cfg['weight_obj'], cfg['weight_obj']], dtype=torch.float32).to(device)
    # if cfg['loss'] == "NLL":
    #     criterion = torch.nn.NLLLoss(weight=weight) #()
    # # elif cfg['loss'] == "CAT":
    # #     criterion = torch.nn.CategoricalCrossEntropyLoss(weight=weight)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss(weight=weight) #NLLLoss()

    if cfg['continue_training'] == True:
        cnt = "continuation"
        model.load_state_dict(torch.load(cfg['ckp_path'], weights_only=True))
        print('continue training..')
    else:
        cnt = 'from_scratch'
        print("start training..")
    
    train_loader = DataLoader(training_set, batch_size=cfg['batch_size'], shuffle=True)
    valid_loader = DataLoader(validation_set, batch_size=cfg['batch_size'], shuffle=True)
    # test_loader = DataLoader(test_set, shuffle=True)

    if cfg['continue_training'] == True:
        cnt = "continuation"
        model.load_state_dict(torch.load(cfg['ckp_path'], weights_only=True))
    else:
        cnt = 'from_scratch'

    # saving folders
    model_name_save = f"sand_scene-{task}-net_{cfg['model']}-based_trained_on_{dataset_name}_split_{split_num}_using_loss{cfg['loss']}_for{cfg['epochs']}epochs_{cnt}_bs_{cfg['batch_size']}_noise{cfg['add_noise']}"
    os.makedirs(os.path.join(cfg['models_path'], model_name_save), exist_ok=True)

    best_loss = cfg['batch_size']   
    best_model_name = ""
    valid_acc_threshold = 0
    nothing_happening = 0
    history = {'epoch': [], 'loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(0, cfg['epochs']):
        correct = 0
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            # ADD NOISE
            if cfg['add_noise'] == True:
                data = add_noise(data, cfg['noise_strength'])
            data.to(device)
            out = model(data.x, data.edge_index)    # Perform a single forward pass.
            loss = criterion(out, data.y)           # Compute the loss.
            loss.backward()                         # Derive gradients.
            optimizer.step()                        # Update parameters based on gradients.
            optimizer.zero_grad()                   # Clear gradients.
            # correct += np.sum(out == data.y) / out.shape[0]
            pred = out.argmax(dim=1)                            # Use the class with highest probability.
            label_class = data.y.argmax(dim=1)
            correct += ((pred == label_class).sum() / out.shape[0]).item()

        if (epoch+1) % cfg['evaluate_and_print_each'] == 0:
            print("_" * 65)
            vcorrect = 0
            model.eval()
            for vdata in valid_loader:
                if cfg['add_noise'] == True:
                    vdata = add_noise(vdata, cfg['noise_strength'])
                vdata.to(device)
                vout = model(vdata.x, vdata.edge_index)    # Perform a single forward pass.
                loss = criterion(vout, vdata.y)                       # Compute the loss.                           
                vpred = vout.argmax(dim=1)
                vlabel_class = vdata.y.argmax(dim=1)
                vcorrect += ((vpred == vlabel_class).sum() / vout.shape[0]).item()  # Check against ground-truth labels.
                
            valid_acc = (vcorrect / len(valid_loader.dataset))
            train_acc = (correct / len(train_loader.dataset))
            history['loss'].append(loss.item())
            history['train_acc'].append(train_acc)
            history['val_acc'].append(valid_acc)
            history['epoch'].append(epoch+1)
            
            print(f'Epoch: {(epoch+1):05d}, Loss: {(loss.item() / len(train_loader.dataset)):.4f}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}')
            if valid_acc > valid_acc_threshold:
                valid_acc_threshold = valid_acc
                nothing_happening = 0
            else:
                nothing_happening += 1

        if nothing_happening > cfg['patience']:
            print("early stopping!")
            break

        if loss.item() < best_loss:
            torch.save(model.state_dict(), os.path.join(cfg['models_path'], model_name_save, 'best.pth'))
    
    print("#" * 50)

    # SAVING THE WEIGHTS
    torch.save(model.state_dict(), os.path.join(cfg['models_path'], model_name_save, 'last.pth'))
    # load bwith 
    # model = detectionGCN(*args, **kwargs)
    # model.load_state_dict(torch.load('last.pth', weights_only=True))

    # SAVE THE FULL MODEL
    if cfg['save_full_model'] == True:
        torch.save(model, os.path.join(cfg['models_path'], model_name_save, 'model.pt'))
        # load with 
        # model = torch.load('model.pt')

    # SCRIPTED
    if cfg['save_scripted_model'] == True:
        model_scripted = torch.jit.script(model)
        model_scripted.save(os.path.join(cfg['models_path'], model_name_save, 'model_scripted.pt'))
        # load with 
        # model = torch.jit.load('model_scripted.pt')

    cfg['dataset_name'] = dataset_name
    cfg['model_folder'] = model_name_save
    cfg['last_model_path'] = os.path.join(cfg['models_path'], model_name_save, 'last.pth')
    cfg['best_model_path'] = os.path.join(cfg['models_path'], best_model_name, 'best.pth')

    res_cfg_path = os.path.join(cfg['models_path'], model_name_save, 'config.yaml')
    with open(res_cfg_path, 'w') as yf:
        yaml.dump(cfg, yf)

    hdf = pd.DataFrame()
    hdf['epoch'] = history['epoch']
    hdf['loss'] = history['loss']
    hdf['training accuracy'] = history['train_acc']
    hdf['validation accuracy'] = history['val_acc']
    hdf.to_csv(os.path.join(cfg['models_path'], model_name_save, 'training_history.csv'))
  
    # shutil.copy(cfg_file_path, os.path.join(cfg['models_path'], f"{model_name_save}_config.yaml"))
    print(f"saved {model_name_save}")
    print(f"For inference, run:")
    print(f"\npython sand_scene_evaluate.py {res_cfg_path}\n")
    
    if cfg['show_results'] == True:
        print(f"showing {cfg['how_many']} results..")
        model.eval()
        
        idx_to_show = np.linspace(0, len(validation_set)-1, cfg['how_many']).astype(int)
        for j in idx_to_show:
            data = validation_set[j]
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
            # breakpoint()
