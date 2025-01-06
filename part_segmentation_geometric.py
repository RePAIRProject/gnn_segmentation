import os, json, pdb
import open3d as o3d
import numpy as np

def compute_roughness(point, neighbours, normal_point, normals_neighbours,
                      method="huang2006"):

    if method == 'huang2006':
        numerator = np.square(normal_point - normals_neighbours)
        denominator = np.square(point - neighbours)
        roughness = 1 / len(neighbours) * \
            np.sum(numerator) / np.sum(denominator)
    else:
        n_normals_avg = np.mean(normals_neighbours)
        roughness = np.sum(np.abs(n_normals_avg - normal_point))
    return roughness

def segment_using_normals(pcl, neighbours=50, threshold=0.005):
    """
    segmenting using formula inspired from surface roughness / local bending energy, eq. (5)
    from Huang 2006 "Reassembling fractured objects by geometric matching" paper.
    """
    if len(pcl.normals) == 0:
        print("found no normals, estimating them now with default parameters")
        pcl.estimate_normals()
    if len(pcl.colors) == 0:
        print("# WARNING: no colors, adding them - is this correct?")
        pcl.colors = o3d.utility.Vector3dVector(np.ones((len(pcl.points), 3)))

    segmentation_classes = np.zeros((1, np.asarray(pcl.points).shape[0]))
    print(f"segmenting with {neighbours} neighbours, this may take a while..")
    pcl_KDTree = o3d.geometry.KDTreeFlann(pcl)
    for j in range(len(np.asarray(pcl.points))):
        [k, idx, _] = pcl_KDTree.search_knn_vector_3d(
            pcl.points[j], neighbours)
        # normals of the neighbours, using the average
        # should we use average of differences for curved surfaces?
        n_normals = np.asarray(pcl.normals)[idx[1:], :]
        neighbouring_points = np.asarray(pcl.points)[idx[1:], :]
        roughness = compute_roughness(
            pcl.points[j], neighbouring_points, pcl.normals[j], n_normals)
        if roughness < threshold:
            pcl.colors[j] = [0, 255, 0]
            segmentation_classes[0, j] = 0
        else:
            pcl.colors[j] = [255, 0, 0]
            segmentation_classes[0, j] = 1
    return segmentation_classes

if __name__ == '__main__':

    # 1. read pcl and labels
    group = 15
    RePAIR_dataset = "/home/palma/Unive/RePAIR/Datasets/RePAIR_dataset"
    #RePAIR_dataset = "/Users/Palma/Documents/Projects/Unive/RePAIR/Datasets/RePAIR_dataset"
    group_folder = os.path.join(RePAIR_dataset, f"group_{group}")
    processed_folder = os.path.join(group_folder, "processed")
    labels_folder = os.path.join(os.getcwd(), "labels")

    rpf_names = ['RPf_00097', 'RPf_00094', 'RPf_00103']
    results = {}
    for rpf_name in rpf_names:

        pcl = o3d.io.read_point_cloud(
            os.path.join(processed_folder, f"{rpf_name}.ply"),
            print_progress=True)
        segmented = segment_using_normals(pcl)
        labels = np.loadtxt(os.path.join(labels_folder, f"{rpf_name}.txt"))
        errors = np.abs(segmented-labels)
        pdb.set_trace()
        o3d.visualization.draw_geometries([pcl])
        error = np.sum(errors) / len(labels)
        error_intact = np.sum(errors * (labels == 0)) / np.sum(labels==0)
        error_broken = np.sum(errors * (labels == 1)) / np.sum(labels==1)
        results[rpf_name] = {
            'acc' : 1-error,
            'acc_intact' : 1-error_intact,
            'acc_broken' : 1-error_broken
        }
        print('acc', 1-error)
    output_folder = os.path.join(os.getcwd(), 'results_classification')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    with open(os.path.join(output_folder,
        f"results_geometric_classification.json"), 'w') as jf:
        json.dump(results, jf, indent=3)
