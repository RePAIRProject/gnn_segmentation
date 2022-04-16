# gnn_segmentation
A first attempt at using GNN (in particular using the GCN layers) to segment fragments.

## Node Classification using GCN
Started using GCN layers for a supervised task.
Labels are obtained from segments.ai labeling tools.

The code is in `node_classification.py`. Dataset root folder has to be changed (line 116) and the other files (processed pointclouds) are assumed to be in the folder structure described in the deliverable.

#### Results
A first run has been made on 6 fragments of group 15. Results are good, but not enough data to guarantee that results are actually significant. Issue is, labeling is slow, so for the moment we do not have more labeled data.

The following results were achieved:

| Training set | Accuracy | Acc. Intact | Acc. Broken | Test Set | Accuracy | Acc. Intact | Acc. Broken |
|:------------:|:--------:|:-----------:|:-----------:|:--------:|:--------:|:-----------:|:-----------:|
| "RPf_00109","RPf_00096","RPf_00095" | 0.95627 |  0.98982 | 0.87532 | "RPf_00097","RPf_00094","RPf_00103" | 0.97214 | 0.99241 | 0.92772 |

With a loss of `0.12146` after 1501 epochs on a Nvidia GPU RTX 3060.

## Weak supervised segmentation
The next approach to start with.
