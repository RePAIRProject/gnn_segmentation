# gnn_segmentation
A first attempt at using GNN (in particular using the GCN layers) to segment fragments.

## Installation 
Required:
```
pytorch
pytorch_geometric
numpy
matplotlib
open3d
sklearn
```

## Node Classification using GCN
Started using GCN layers for a supervised task.
Labels are obtained from segments.ai labeling tools.

The code is in `node_classification_gnn.py`. Dataset root folder has to be changed (line 116) and the other files (processed pointclouds) are assumed to be in the folder structure described in the deliverable.

### Results on binary segmentation
A first run has been made on 6 fragments of group 15.
Results are good, but not enough data to guarantee that results are actually significant.
Plus all 6 pointclouds were labeled from a single person, introducing some bias on the labels.
Labeling is slow, so for the moment we do not have more labeled data.

The following results were achieved:

| Set | Accuracy | Acc. Intact | Acc. Broken | Files |
|:---:|:--------:|:-----------:|:-----------:|:-----:|
| Train | 0.95627 |  0.98982 | 0.87532 | "RPf_00109","RPf_00096","RPf_00095" |
| Test | 0.97214 | 0.99241 | 0.92772 | "RPf_00097","RPf_00094","RPf_00103" |

With a loss of `0.12146` after 1501 epochs on a Nvidia GPU RTX 3060.

#### Baseline
To have an idea about the task, we have computed results using a simple baseline method.
We take inspiration from the paper *Reassembling fractured objects by geometric matching*, Huang et al. Siggraph 2006 and use the Eq (5) of that paper to compute surface roughness, then we can classify the points as belonging to intact or broken surface based on that value.

The code is in `node_classification_geometric.py`.
Please note that the labels were done by hand and 'seen' for the training, but they do not necessarily respect the geometrical classification we are doing here, so there is a slight bias in favour of the neural network when evaluating using these labels.
However, even such simple metrics yield okay results, giving us a hint about the difficulty of the task in general (this may change when we move to multi-label classification, even distinguishing bottom to top may be harder for the geometrical approach):

| Fragment | Accuracy | Acc. Intact | Acc. Broken |
|:--------:|:--------:|:-----------:|:-----------:|
| averaged  | 0.86073 | 0.86265 | 0.84959 |
| RPf_0097 | 0.82433 | 0.84089 | 0.78698 |
| RPf_0094 | 0.87539 | 0.85453 | 0.91485 |
| RPf_0094 | 0.88248 | 0.89254 | 0.84695 |

These results were obtained using default values (50 neighbours for roughness computations).
Here averaged is on the three files ("RPf_00097","RPf_00094","RPf_00103") used also in the "test" set, so the comparison is on the same fragments.

## Weak supervised segmentation
The next approach to start with.


## Workflow for 3D 

Exploring two options

### 1. Step by Step 
- a. Detection (given a 3D scene, binary classification, `background` or `fragment`)
- b. Recognition (assuming we have the database of 3D fragments)
    - b1. Recognition as Retrieval (given a 3D fragment, find the *most similar* in the db)
    - b2. Recognition as Classification (given a 3D fragment, classify it with its ID)

### 2. End-to-end
Train a network to directly *detect* and *classify* the objects (assign the correct IDs)