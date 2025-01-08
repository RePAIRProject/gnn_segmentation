import os 
import yaml 
from utils.dataset import dataset_sand_scene
import numpy as np
import pickle 
from random import shuffle

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
    dataset_15 = dataset_sand_scene(cfg)
    print("\ndone with group 15, now 29")
    cfg['dataset_root'] = os.path.join(dataset_root, 'group_0029')
    dataset_29 = dataset_sand_scene(cfg)
    # SPLIT
    print("\nDone\nSplitting..")
    # breakpoint()
    train_split_15 = np.round(len(dataset_15) * cfg['train_split']).astype(int)
    validation_split_15 = np.round(train_split_15 + len(dataset_15) * cfg['val_split']).astype(int)
    train_split_29 = np.round(len(dataset_29) * cfg['train_split']).astype(int)
    validation_split_29 = np.round(train_split_29 + len(dataset_29) * cfg['val_split']).astype(int)

    # train_dataset_15 = dataset_15[:train_test_split]
    # test_dataset_15 = dataset_15[train_test_split:]
    # train_dataset_29 = dataset_29[:train_test_split]
    # test_dataset_29 = dataset_29[train_test_split:]
    # train_dataset = train_dataset_15 + train_dataset_29
    # test_dataset = test_dataset_15 + test_dataset_29
    # SAVE 
    print("Saving..")
    dataset_name = dataset_root.split('/')[-1]
    dataset_folder = os.path.join('data', f'dataset_from_{dataset_name}_for_{task}_xyzrgb')
    os.makedirs(dataset_folder, exist_ok=True)
    for i in range(cfg['num_splits']):
        shuffle(dataset_15)
        shuffle(dataset_29)
        training_set_15 = dataset_15[:train_split_15]
        validation_set_15 = dataset_15[train_split_15:validation_split_15]
        test_set_15 = dataset_15[validation_split_15:]
        training_set_29 = dataset_29[:train_split_29]
        validation_set_29 = dataset_29[train_split_29:validation_split_29]
        test_set_29 = dataset_29[validation_split_29:]

        training_set = training_set_15 + training_set_29
        validation_set = validation_set_15 + validation_set_29
        test_set = test_set_15 + test_set_29
        # SAVE 
        print(f"Saving split {i+1}, training: {len(training_set)}, valid: {len(validation_set)}, test: {len(test_set)}..", end='\r')
        with open(os.path.join(dataset_folder, f'training_set_split_{i+1}'), 'wb') as file: 
            pickle.dump(training_set, file)
        with open(os.path.join(dataset_folder, f'validation_set_split_{i+1}'), 'wb') as file: 
            pickle.dump(validation_set, file)
        with open(os.path.join(dataset_folder, f'test_set_split_{i+1}'), 'wb') as file: 
            pickle.dump(test_set, file)

    # with open(os.path.join('data', f'{task}_sand_training_set'), 'wb') as file: 
    #     pickle.dump(train_dataset, file)
    # with open(os.path.join('data', f'{task}_sand_test_set'), 'wb') as file: 
    #     pickle.dump(test_dataset, file)