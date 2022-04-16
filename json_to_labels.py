import open3d as o3d
import os
import numpy as np
import pdb
import json

if __name__ == '__main__':
    """
    It converts the json file from segments.ai to a .txt file with the labels (0,1).
    It also reads each .ply to check labels and pcl size are matching (assertion on line 35)
    """
    # segmentation labels
    group = 15
    RePAIR_dataset = "/home/palma/Unive/RePAIR/Datasets/RePAIR_dataset"
    group_folder = os.path.join(RePAIR_dataset, f"group_{group}")
    processed_folder = os.path.join(group_folder, "processed")
    labeled_folder = os.path.join(group_folder, "labeled")
    labels_folder = os.path.join(os.getcwd(), "labels")
    json_path = os.path.join(labels_folder, f"Group_{group}-v0.2.json")
    with open(json_path, 'r') as j:
        contents = json.loads(j.read())
    # samples which have been labeled
    samples_data = contents['dataset']['samples']
    for sample in samples_data:
        # check which piece is
        s_name = sample['name']
        rpf_name = s_name[:9]
        # get labels
        if sample['labels']['ground-truth']:
            s_labels = np.asarray(
                sample['labels']['ground-truth']['attributes']['point_annotations'])
            # read original pcl
            orig_path = os.path.join(processed_folder, f"{rpf_name}.ply")
            orig_pcl = o3d.io.read_point_cloud(orig_path)
            assert(len(orig_pcl.points) == len(s_labels)), "pointcloud and labels not matching!"
            labels_path = os.path.join(labels_folder, f"{rpf_name}.txt")
            np.savetxt(labels_path, s_labels, fmt='%d')
