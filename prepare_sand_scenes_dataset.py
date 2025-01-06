import os 
import yaml 
from utils.dataset import dataset_v3
import numpy as np
import pickle 

if __name__ == '__main__':

    task = 'detection' # 'recognition' or 'detection'
    print("#" * 50)
    print(f"\nPreparing dataset for {task}\n")
    cfg_file_path = os.path.join('configs', f'cfg_{task[:3]}.yaml')
    with open(cfg_file_path, 'r') as yf:
        cfg = yaml.safe_load(yf)

    # DATASET
    print("Reading the data..")
    dataset_root = cfg['dataset_root']
    cfg['dataset_root'] = os.path.join(dataset_root, 'group_0015')
    dataset_15 = dataset_v3(cfg)
    cfg['dataset_root'] = os.path.join(dataset_root, 'group_0029')
    dataset_29 = dataset_v3(cfg)
    # SPLIT
    print("Splitting..")
    train_test_split = np.round(cfg['dataset_max_size'] * cfg['train_test_split']).astype(int)
    train_dataset_15 = dataset_15[:train_test_split]
    test_dataset_15 = dataset_15[train_test_split:]
    train_dataset_29 = dataset_29[:train_test_split]
    test_dataset_29 = dataset_29[train_test_split:]
    train_dataset = train_dataset_15 + train_dataset_29
    test_dataset = test_dataset_15 + test_dataset_29
    # SAVE 
    print("Saving..")
    with open(os.path.join('data', f'{task}_sand_training_set'), 'wb') as file: 
        pickle.dump(train_dataset, file)
    with open(os.path.join('data', f'{task}_sand_test_set'), 'wb') as file: 
        pickle.dump(test_dataset, file)