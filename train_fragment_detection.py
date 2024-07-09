import torch 
from dataset import prepare_dataset_detection, dataset_from_pcl
from net import GAT, GCN_tutorial
from torch_geometric.loader import DataLoader
from train_test_util import predict, training_loop_one_epoch, test_with_loader, \
    show_results
import os, json
import open3d as o3d 
import numpy as np 

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    dataset = prepare_dataset_detection('/home/palma/Datasets/3D_detection_fixed_camera/group_0016', \
         dataset_max_size=5, k=5, use_color=False)
    #breakpoint()
    # dataset = dataset_from_pcl('/home/palma/Datasets/segmented_pcl', \
    #     dataset_max_size=25, k=5)
    # prepare the model
    input_features = 3
    hidden_channels = 32
    output_classes = 8
    print(f"GCN Model with: \
          {input_features} input features, \
          {hidden_channels} hidden_channels and \
          {output_classes} output_classes")
    # 4. create GCN model
    model = GAT(input_features=input_features,
                hidden_channels=hidden_channels,
                output_classes=output_classes)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    weight = torch.tensor([1, 25, 25, 25, 25, 25, 25, 25], dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight) #NLLLoss()

    print("start training..")
    EPOCHS = 200
    test_acc = 0.0
    acc_intact = 0.0
    acc_broken = 0.0

    train_test_split = 4
    train_dataset = dataset[:train_test_split]
    test_dataset = dataset[train_test_split:]
    # train_files = names[:train_test_split]
    # test_files = names[train_test_split:]
    train_loader = DataLoader(train_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, shuffle=False)
    #breakpoint()

    model.train()

    for epoch in range(1, EPOCHS):
        # loss = training_loop_one_epoch(model, train_loader, criterion, optimizer, device)
        for data in train_loader:  # Iterate in batches over the training dataset.
            data.to(device)
            out = model(data.x, data.edge_index)  # Perform a single forward pass.
            loss = criterion(out, data.y-1)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

        # print(loss.item())

        if epoch % 1 == 0:
            #pdb.set_trace()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    #     if epoch > 0 and epoch % 10 == 0:
    #         #pdb.set_trace()
    #         train_avg_correct_points, train_acc_areas, train_acc_ratio = test_with_loader(model, device, train_loader)
    #         test_avg_correct_points, test_acc_areas, test_acc_ratio = test_with_loader(model, device, test_loader)
    #         print('Training:')
    #         print(f'average correct points: {train_avg_correct_points:.4f}, accuracy ratio: {train_acc_ratio:.4f}')
    #         print(f'accuracy on intact surfaces: {train_acc_areas[0]:.4f}, accuracy on broken surfaces: {train_acc_areas[1]:.4f}')
    #         print('Testing:')
    #         print(f'average correct points: {test_avg_correct_points:.4f}, accuracy ratio: {test_acc_ratio:.4f}')
    #         print(f'accuracy on intact surfaces: {test_acc_areas[0]:.4f}, accuracy on broken surfaces: {test_acc_areas[1]:.4f}')
    
    # results = {
    #     'loss': loss.item(),
    #     'training': {
    #         'acc_ratio': train_acc_ratio,
    #         'acc_intact': train_acc_areas[0],
    #         'acc_broken': train_acc_areas[1],
    #         # 'files': train_files
    #     },
    #     'test': {
    #         'acc_ratio': test_acc_ratio,
    #         'acc_intact': test_acc_areas[0],
    #         'acc_broken': test_acc_areas[1],
    #         # 'files': test_files
    #     },
    #     'epochs': EPOCHS,
    #     'device': str(device)
    # }
    # output_folder = os.path.join(os.getcwd(), 'results_detection')
    # if not os.path.exists(output_folder):
    #     os.mkdir(output_folder)
    # with open(os.path.join(output_folder,
    #     f"results_after_{epoch}_epochs.json"), 'w') as jf:
    #     json.dump(results, jf, indent=3)
    # if test_acc > 0 and acc_broken > 0.5:
    #     np.savetxt(os.path.join(output_folder,
    #         f"pred_{rpf_name}_after_{epoch}_epochs.txt"), pred.numpy())

    model.eval()
    breakpoint()
    for j in range(2, 3):
        pred = predict(model, dataset[j], device) # pred returned is already .cpy().numpy()
        pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(dataset[j].pos.cpu().numpy()))
        show_results(pred, pcl)

    breakpoint()
