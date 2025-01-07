import os 
import yaml 
from utils.dataset import dataset_v3, dataset_of_fragments
import numpy as np
import pickle 
from random import shuffle

if __name__ == '__main__':

    task = 'recognition' # 'recognition' or 'detection'
    
    print("#" * 50)
    print("GROUP 15")
    print("#" * 50)
    print(f"\nPreparing dataset for {task}\n")
    cfg_file_path = os.path.join('configs', f'cfg_{task[:3]}.yaml')
    with open(cfg_file_path, 'r') as yf:
        cfg = yaml.safe_load(yf)
    # DATASET
    print("Reading the data..")
    dataset_root = cfg['dataset_root']
    dataset_name = dataset_root.split('/')[-1]
    # force group and classes 
    cfg['group'] = 15
    group = cfg['group']
    cfg['num_classes'] = 17
    cfg['dataset_root'] = os.path.join(dataset_root, f'group_{group:04d}')
    dataset = dataset_of_fragments(cfg)
    # SPLIT
    print("\nSplitting..")
    dataset_folder = os.path.join('data', f'dataset_from_{dataset_name}_group_{group}_fragments_{task}')
    os.makedirs(dataset_folder, exist_ok=True)
    train_split = np.round(len(dataset) * cfg['train_split']).astype(int)
    val_split = np.round(train_split + len(dataset) * cfg['val_split']).astype(int)
    for i in range(cfg['num_splits']):
        shuffle(dataset)
        training_set = dataset[:train_split]
        validation_set = dataset[train_split:val_split]
        test_set = dataset[val_split:]
    
        # SAVE 
        print(f"Saving split {i+1}, training: {len(training_set)}, valid: {len(validation_set)}, test: {len(test_set)}..", end='\r')
        with open(os.path.join(dataset_folder, f'training_set_split_{i+1}'), 'wb') as file: 
            pickle.dump(training_set, file)
        with open(os.path.join(dataset_folder, f'validation_set_split_{i+1}'), 'wb') as file: 
            pickle.dump(validation_set, file)
        with open(os.path.join(dataset_folder, f'test_set_split_{i+1}'), 'wb') as file: 
            pickle.dump(test_set, file)
    print("\nDone\n")

    print("#" * 50)
    print("GROUP 29")
    print("#" * 50)
    print(f"\nPreparing dataset for {task}\n")
    cfg_file_path = os.path.join('configs', f'cfg_{task[:3]}.yaml')
    with open(cfg_file_path, 'r') as yf:
        cfg = yaml.safe_load(yf)
    # DATASET
    print("Reading the data..")
    cfg['group'] = 29
    group = cfg['group']
    cfg['num_classes'] = 5
    dataset_root = cfg['dataset_root']
    cfg['dataset_root'] = os.path.join(dataset_root, f'group_{group:04d}')
    dataset = dataset_of_fragments(cfg)
    # SPLIT
    print("\nSplitting..")
    dataset_folder = os.path.join('data', f'dataset_from_{dataset_name}_group_{group}_fragments_{task}')
    os.makedirs(dataset_folder, exist_ok=True)
    train_split = np.round(len(dataset) * cfg['train_split']).astype(int)
    val_split = np.round(train_split + len(dataset) * cfg['val_split']).astype(int)
    for i in range(cfg['num_splits']):
        shuffle(dataset)
        training_set = dataset[:train_split]
        validation_set = dataset[train_split:val_split]
        test_set = dataset[val_split:]
        # SAVE 
        print(f"Saving split {i+1}, training: {len(training_set)}, valid: {len(validation_set)}, test: {len(test_set)}..", end='\r')
        with open(os.path.join(dataset_folder, f'training_set_split_{i+1}'), 'wb') as file: 
            pickle.dump(training_set, file)
        with open(os.path.join(dataset_folder, f'validation_set_split_{i+1}'), 'wb') as file: 
            pickle.dump(validation_set, file)
        with open(os.path.join(dataset_folder, f'test_set_split_{i+1}'), 'wb') as file: 
            pickle.dump(test_set, file)
    print("\nDone\n")