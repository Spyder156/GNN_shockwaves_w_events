import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GNNModelWithNodeEvents(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, event_dim):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        
        # Process event features
        self.event_fc = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Combine GNN and event embeddings
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, data):
        x, edge_index, event_features = data.x, data.edge_index, data.event_features
        x = self.gat1(x, edge_index).relu()
        x = self.gat2(x, edge_index).relu()
        
        # Process event features for each node
        event_embeddings = self.event_fc(event_features)
        
        # Combine GNN node embeddings and event embeddings
        combined_embeddings = torch.cat([x, event_embeddings], dim=-1)
        output = self.fc(combined_embeddings)
        return output
