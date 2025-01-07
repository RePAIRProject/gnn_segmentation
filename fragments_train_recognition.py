import torch 
from utils.dataset import prepare_dataset_detection, dataset_from_pcl, dataset_v2, dataset_v3
from network.gnns import GCN, GAT, recognitionGCN
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
    
    print("#" * 50)
    print(f"\nTraining for {task}\n")
    cfg_file_path = os.path.join('configs', f'cfg_{task[:3]}.yaml')
    with open(cfg_file_path, 'r') as yf:
        cfg = yaml.safe_load(yf)
    
    # adjust for this group
    group = cfg['group']
    dataset_name = cfg['dataset_root'].split('/')[-1]
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
    # data/dataset_from_sand_detection_dataset_bb_yolo_1000scenes_group_29_fragments_recognition
    dataset_path = os.path.join('data', f'dataset_from_{dataset_name}_group_{group}_fragments_{task}')
    print('using training data in', dataset_path)
    with open(os.path.join(dataset_path, 'training_set_split_1'), 'rb') as training_set_file: 
        training_set = pickle.load(training_set_file)
    with open(os.path.join(dataset_path, 'validation_set_split_1'), 'rb') as valid_set_file: 
        validation_set = pickle.load(valid_set_file)
    with open(os.path.join(dataset_path, 'test_set_split_1'), 'rb') as test_set_file: 
        test_set = pickle.load(test_set_file)

    # show data
    # for k in range(0, 10):
    #     pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(train_dataset[k].x[:,:3]))
    #     pcl.paint_uniform_color((0,0,1))
    #     name = f"object of class {int(train_dataset[k].y.argmax(dim=1))}"
    #     o3d.visualization.draw_geometries([pcl], window_name = name)
    # breakpoint()

    print('model..')
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
                            output_classes=output_classes)
    else:
        print("WHICH MODEL?")

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=5e-4)

    # WEIGHTS (for imbalanced datasets)
    if cfg['task'] == 'detection':
        weight = torch.tensor([1, cfg['weight_obj']], dtype=torch.float32).to(device)
    elif cfg['task'] == 'recognition':
        weights = np.ones((cfg['num_classes'])) * cfg['weight_obj']
        # weights[0] /= 5
        weight = torch.tensor(weights, dtype=torch.float32).to(device)

    # LOSS
    if cfg['loss'] == "NLL":
        criterion = torch.nn.NLLLoss(weight=weight) #()
    else:
        criterion = torch.nn.CrossEntropyLoss() #NLLLoss()

    print("start training..")
    EPOCHS = cfg['epochs']
    test_acc = 0.0
    acc_intact = 0.0
    acc_broken = 0.0
    train_loader = DataLoader(training_set, batch_size=cfg['batch_size'], shuffle=True)
    valid_loader = DataLoader(validation_set, batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False)

    if cfg['continue_training'] == True:
        cnt = "continuation"
        model.load_state_dict(torch.load(cfg['ckp_path'], weights_only=True))
    else:
        cnt = 'from_scratch'
    best_loss = 1    
    model_name_save = f"fragment-{task}-net_{model_name}-based_trained_on_{dataset_name}-group_{group:04d}_using_loss{cfg['loss']}_for{EPOCHS}epochs_{cnt}_bs_{cfg['batch_size']}"
    os.makedirs(os.path.join(cfg['models_path'], model_name_save), exist_ok=True)
    best_model_name = ""
    valid_acc_threshold = 0
    nothing_happening = 0
    # TRAINING
    model.train()
    for epoch in range(0, EPOCHS):
        correct = 0
        losses = 0
        # loss = training_loop_one_epoch(model, train_loader, criterion, optimizer, device)
        for data in train_loader:  # Iterate in batches over the training dataset.
            data.to(device)
            # print(model, data)
            out = model(data.x, data.edge_index, data.batch)    # Perform a single forward pass.
            loss = criterion(out, data.y)                       # Compute the loss.
            losses+=loss                            
            loss.backward()                                     # Derive gradients.
            optimizer.step()                                    # Update parameters based on gradients.
            optimizer.zero_grad()                               # Clear gradients.
            pred = out.argmax(dim=1)                            # Use the class with highest probability.
            # print(out)
            label_class = data.y.argmax(dim=1)
            correct += int((pred == label_class).sum())  # Check against ground-truth labels.
        # print(loss.item())

        if (epoch+1) % cfg['evaluate_and_print_each'] == 0:
            print("#" * 50)
            print("EVALUATION")
            print(f'Epoch: {(epoch+1):03d}, Loss: {(losses / len(train_loader.dataset)):.4f}')
            vcorrect = 0
            for vdata in valid_loader:
                data.to(device)
                out = model(data.x, data.edge_index, data.batch)    # Perform a single forward pass.
                loss = criterion(out, data.y)                       # Compute the loss.
                losses+=loss                            
                loss.backward()                                     # Derive gradients.
                optimizer.step()                                    # Update parameters based on gradients.
                optimizer.zero_grad()                               # Clear gradients.
                pred = out.argmax(dim=1)
                label_class = data.y.argmax(dim=1)
                vcorrect += int((pred == label_class).sum())  # Check against ground-truth labels.
                valid_acc = (vcorrect / len(valid_loader.dataset))
            print(f'Train Acc: {(correct / len(train_loader.dataset)):.4f}, Valid Acc: {valid_acc:.4f}')
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
        if loss.item() < best_loss:
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
        
    # shutil.copy(cfg_file_path, os.path.join(cfg['models_path'], f"{model_name_save}_config.yaml"))
    print(f"saved {model_name_save}")
    print(f"For inference, run:")
    print(f"\npython fragments_evaluate.py {res_cfg_path}\n")
    
    if cfg['show_results'] == True:
        print(f"showing {cfg['how_many']} results..")
        model.eval()
        
        # idx_to_show = np.linspace(0, len(test_dataset)-1, cfg['how_many']).astype(int)
        counter = 0
        for data in test_loader:
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
            # print(f"predicted: {pred_class.item()} // correct: {label_class.item()} \n pred_values({out.cpu().detach().numpy()})")
            # pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(test_dataset[j].pos.cpu().numpy()))
            # print('pred')
            # show_results(pred, pcl, window_name=f"Prediction Scene {j}")
            # print('gt')
            # # breakpoint()
            # labels = (test_dataset[j].y).cpu().numpy()
            # show_results(labels, pcl, window_name=f"Ground Truth Scene {j}")
            # breakpoint()
