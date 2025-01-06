import torch 
from torch_geometric.nn import GATConv, GCNConv, GraphConv, global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear

class recognitionGCN(torch.nn.Module):

    def __init__(self, input_features, hidden_channels, output_classes):
        super().__init__()

        self.conv1 = GCNConv(input_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

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

