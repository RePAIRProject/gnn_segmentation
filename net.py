import torch 
from torch_geometric.nn import GATConv, GCNConv, GraphConv
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, input_features, hidden_channels, output_classes):
        super().__init__()

        self.conv1 = GCNConv(input_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels//2)
        self.conv3 = GCNConv(hidden_channels//2, output_classes)

    def forward(self, x, edge_index):
        # breakpoint()
        x = self.conv1(x, edge_index)
        x = x.relu()
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        #x = F.dropout(x, p=0.5, training=self.training)
        #breakpoint()
        #if x.shape[0] != 124458:
        #print(f"x: {x.shape}, edge_index: {edge_index.shape}")
        x = self.conv3(x, edge_index)
        return x

class GCN_tutorial(torch.nn.Module):
    def __init__(self, input_features, hidden_channels, output_classes):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(input_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_classes)
        self.softmax = F.Softmax()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        #x = self.softmax()
        return x

#model = GCN(hidden_channels=16)
#print(model)