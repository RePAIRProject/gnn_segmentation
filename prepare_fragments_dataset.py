import os 
import yaml 
from utils.dataset import dataset_v3, dataset_of_fragments
import numpy as np
import pickle 

if __name__ == '__main__':

    task = 'recognition' # 'recognition' or 'detection'
    group = 15
    print("#" * 50)
    print("GROUP 15")
    print("#" * 50)
    print(f"\nPreparing dataset for {task}\n")
    cfg_file_path = os.path.join('configs', f'cfg_{task[:3]}.yaml')
    with open(cfg_file_path, 'r') as yf:
        cfg = yaml.safe_load(yf)
    cfg['num_frags'] = 17
    # DATASET
    print("Reading the data..")
    dataset_root = cfg['dataset_root']
    cfg['dataset_root'] = os.path.join(dataset_root, f'group_{group:04d}')
    dataset = dataset_of_fragments(cfg)
    # SPLIT
    print("\nSplitting..")
    train_test_split = np.round(cfg['dataset_max_size'] * cfg['train_test_split']).astype(int)
    train_dataset = dataset[:train_test_split]
    test_dataset = dataset[train_test_split:]
    # SAVE 
    print("Saving..")
    with open(os.path.join('data', f'group_{group}_fragments_{task}_training_set'), 'wb') as file: 
        pickle.dump(train_dataset, file)
    with open(os.path.join('data', f'group_{group}_fragments_{task}_test_set'), 'wb') as file: 
        pickle.dump(test_dataset, file)
    print("Done\n")

    print("#" * 50)
    print("GROUP 29")
    group = 29
    print("#" * 50)
    print(f"\nPreparing dataset for {task}\n")
    cfg_file_path = os.path.join('configs', f'cfg_{task[:3]}.yaml')
    with open(cfg_file_path, 'r') as yf:
        cfg = yaml.safe_load(yf)
    cfg['num_frags'] = 5
    # DATASET
    print("Reading the data..")
    dataset_root = cfg['dataset_root']
    cfg['dataset_root'] = os.path.join(dataset_root, f'group_{group:04d}')
    dataset = dataset_of_fragments(cfg)
    # SPLIT
    print("\nSplitting..")
    train_test_split = np.round(cfg['dataset_max_size'] * cfg['train_test_split']).astype(int)
    train_dataset = dataset[:train_test_split]
    test_dataset = dataset[train_test_split:]
    # SAVE 
    print("Saving..")
    with open(os.path.join('data', f'group_{group}_fragments_{task}_training_set'), 'wb') as file: 
        pickle.dump(train_dataset, file)
    with open(os.path.join('data', f'group_{group}_fragments_{task}_test_set'), 'wb') as file: 
        pickle.dump(test_dataset, file)
    print("Done\n")