import torch 
from torch_geometric.nn import GATConv, GCNConv, GraphConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, input_features, hidden_channels, output_classes):
        super().__init__()

        self.conv1 = GCNConv(input_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels//2)
        self.conv3 = GCNConv(hidden_channels//2, output_classes)

    def forward(self, x, edge_index):
        # breakpoint()
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, input_features, hidden_channels, output_classes):
        super().__init__()

        self.conv1 = GATConv(input_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels//2)
        self.conv3 = GATConv(hidden_channels//2, output_classes)

    def forward(self, x, edge_index):
        # breakpoint()
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x

