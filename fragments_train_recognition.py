import torch 
from utils.dataset import prepare_dataset_detection, dataset_from_pcl, dataset_v2, dataset_v3

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

    task = 'recognition' # 'recognition' or 'detection'
    print("#" * 50)
    print(f"\nTraining for {task}\n")
    cfg_file_path = os.path.join('configs', f'cfg_{task[:3]}.yaml')
    with open(cfg_file_path, 'r') as yf:
        cfg = yaml.safe_load(yf)
    
    # adjust for this group
    group = cfg['group']
    dataset_name = cfg['dataset_root'].split('/')[-1]
    cfg['dataset_root'] = os.path.join(cfg['dataset_root'], f'group_{group:04d}')
    print_parameters(cfg)
   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} to train..")
    print('reading data..')
    # data/dataset_from_sand_detection_dataset_bb_yolo_1000scenes_group_29_fragments_recognition
    dataset_path = os.path.join('data', f'dataset_from_{dataset_name}_group_{group}_fragments_{task}_xyzrgb')
    print('using training data in', dataset_path)
    split_num = 7
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
    
    if cfg['continue_training'] == True:
        cnt = "continuation"
        model.load_state_dict(torch.load(cfg['ckp_path'], weights_only=True))
        print('continue training..')
    else:
        cnt = 'from_scratch'
        print("start training..")
    
    train_loader = DataLoader(training_set, batch_size=cfg['batch_size'], shuffle=True)
    valid_loader = DataLoader(validation_set, batch_size=cfg['batch_size'], shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False)

    # saving folders
    model_name_save = f"fragment-{task}-net_{cfg['model']}-based_trained_on_{dataset_name}_split_{split_num}-group_{group:04d}_using_loss{cfg['loss']}_for{cfg['epochs']}epochs_{cnt}_bs_{cfg['batch_size']}_noise{cfg['add_noise']}"
    os.makedirs(os.path.join(cfg['models_path'], model_name_save), exist_ok=True)

    # keeping tracks
    best_loss = cfg['batch_size']   
    best_model_name = ""
    valid_acc_threshold = 0
    nothing_happening = 0
    history = {'epoch': [], 'loss': [], 'train_acc': [], 'val_acc': []}
    # TRAINING
    for epoch in range(0, cfg['epochs']):
        correct = 0
        losses = 0
        model.train()
        # loss = training_loop_one_epoch(model, train_loader, criterion, optimizer, device)
        for data in train_loader:  # Iterate in batches over the training dataset.
            # ADD NOISE
            if cfg['add_noise'] == True:
                data = add_noise(data, cfg['noise_strength'])
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)    # Perform a single forward pass.
            loss = criterion(out, data.y)                       # Compute the loss.
            losses+=loss                            
            loss.backward()                                     # Derive gradients.
            optimizer.step()                                    # Update parameters based on gradients.
            optimizer.zero_grad()                               # Clear gradients.
            pred = out.argmax(dim=1)                            # Use the class with highest probability.
            label_class = data.y.argmax(dim=1)
            correct += int((pred == label_class).sum())  

        if (epoch+1) % cfg['evaluate_and_print_each'] == 0:
            print("_" * 65)
            
            vcorrect = 0
            model.eval()
            for vdata in valid_loader:
                if cfg['add_noise'] == True:
                    vdata = add_noise(vdata, cfg['noise_strength'])
                vdata.to(device)
                vout = model(vdata.x, vdata.edge_index, vdata.batch)    # Perform a single forward pass.
                loss = criterion(vout, vdata.y)                       # Compute the loss.                           
                vpred = vout.argmax(dim=1)
                vlabel_class = vdata.y.argmax(dim=1)
                vcorrect += int((vpred == vlabel_class).sum())  # Check against ground-truth labels.
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
            #print(nothing_happening)
        if nothing_happening > cfg['patience']:
            print("early stopping!")
            #breakpoint()
            break

        if (loss.item() / len(train_loader.dataset)) < best_loss:
            torch.save(model.state_dict(), os.path.join(cfg['models_path'], model_name_save, 'best.pth'))
    print("#" * 50)
    torch.save(model.state_dict(), os.path.join(cfg['models_path'], model_name_save, 'last.pth'))
    
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
    print(f"\npython fragments_evaluate.py {res_cfg_path}\n")
    
    if cfg['show_results'] == True:
        print(f"showing {cfg['how_many']} results..")
        model.eval()
        
        # idx_to_show = np.linspace(0, len(test_dataset)-1, cfg['how_many']).astype(int)
        counter = 0
        for data in valid_loader:
            counter += 1
            if counter > cfg['how_many']:
                continue
            data.to(device)
            out = model(data.x, data.edge_index, data.batch)  
            pred_class = out.argmax(dim=1)
            label_class = data.y.argmax(dim=1)
            print('-' * 40)
            for pc, cl in zip(pred_class, label_class): 
                print("predicted:", pc.item(), "correct:", cl.item())
