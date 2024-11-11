import torch 
from dataset import prepare_dataset_detection, dataset_from_pcl, dataset_binary_pcl_labels
from net import GCN, GAT
from torch_geometric.loader import DataLoader
from train_test_util import predict, training_loop_one_epoch, test_with_loader, \
    show_results
import os, json
import open3d as o3d 
import numpy as np 
import yaml 
import shutil 

if __name__ == '__main__':

    cfg_name = 'cfg_rp.yaml'
    with open(cfg_name, 'r') as yf:
        cfg = yaml.safe_load(yf)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("\nDEVICE\n", device)
    print("\nDATASET")
    dataset = dataset_binary_pcl_labels(folder = cfg['dataset_root'], k=cfg['k'])
    # prepare_dataset_detection(root_folder = cfg['dataset_root'], \
    #       dataset_max_size=cfg['dataset_max_size'], k=cfg['k'], use_color=cfg['use_color'])
    #breakpoint()
    # dataset = dataset_from_pcl('/home/palma/Datasets/segmented_pcl', \
    #     dataset_max_size=25, k=5)
    # prepare the model
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
        model = GCN(input_features=input_features,
                    hidden_channels=hidden_channels,
                    output_classes=output_classes)
    else:
        print("WHICH MODEL?")

    model.to(device)

    print("\nMODEL")
    print(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=5e-4)
    # weight = torch.tensor([1, cfg['weight_obj'], cfg['weight_obj'], cfg['weight_obj'], cfg['weight_obj']], dtype=torch.float32).to(device)
    weight = torch.tensor([0.65, 0.35], dtype=torch.float32).to(device)
    if cfg['loss'] == "NLL":
        criterion = torch.nn.NLLLoss(weight=weight) #()
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=weight) #weight=weight) #NLLLoss()

    print("\nTRAINING")
    EPOCHS = cfg['epochs']
    test_acc = 0.0
    acc_intact = 0.0
    acc_broken = 0.0
    print(f"Will train for {EPOCHS} epochs")
    train_test_split = np.round(cfg['dataset_max_size'] * cfg['train_test_split']).astype(int)
    train_dataset = dataset[:train_test_split]
    test_dataset = dataset[train_test_split:]
    # train_files = names[:train_test_split]
    # test_files = names[train_test_split:]
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)
    #breakpoint()

    model.train()

    for epoch in range(1, EPOCHS):
        # loss = training_loop_one_epoch(model, train_loader, criterion, optimizer, device)
        for data in train_loader:  # Iterate in batches over the training dataset.
            data.to(device)
            out = model(data.x, data.edge_index)    # Perform a single forward pass.
            loss = criterion(out, data.y)         # Compute the loss.
            loss.backward()                         # Derive gradients.
            optimizer.step()                        # Update parameters based on gradients.
            optimizer.zero_grad()                   # Clear gradients.

        # print(loss.item())

        if epoch % cfg['print_each'] == 0:
            #pdb.set_trace()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    print("\nSAVING")
    os.makedirs(cfg['models_path'], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg['models_path'], f"{model_name}_loss{cfg['loss']}_{EPOCHS}epochs.pth"))
    shutil.copy(cfg_name, os.path.join(cfg['models_path'], f"{model_name}_loss{cfg['loss']}_{EPOCHS}epochs_config.yaml"))
    print(f"saved {model_name}_loss{cfg['loss']}_{EPOCHS}epochs")
    print(f"For inference, run\n")
    print(f"python inference_fragment_detection.py {model_name}_loss{cfg['loss']}_{EPOCHS}epochs\n")
    

    if cfg['show_results'] == True:
        print("\nRESULTS")
        print(f"showing {cfg['how_many']} results..")
        model.eval()
        
        for j in range(0, len(dataset), np.ceil(len(dataset) / cfg['how_many']).astype(int)):
            pred = predict(model, dataset[j], device) # pred returned is already .cpy().numpy()
            pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(dataset[j].pos.cpu().numpy()))
            print('pred')
            show_results(pred, pcl, window_name="Prediction")
            print('gt')
            #breakpoint()
            labels = (dataset[j].y).cpu().numpy()
            show_results(labels, pcl, window_name="Ground Truth")

        breakpoint()
