import torch
from torch_geometric.data import Data
import numpy as np

def create_graph(data_window):
    """
    Create a PyTorch Geometric graph object from stock data.

    Args:
        data_window (pd.DataFrame): A dataframe of stock data for the current window.

    Returns:
        Data: A PyTorch Geometric graph object.
    """
    # Node features
    node_features = torch.tensor(data_window.values, dtype=torch.float).T  # Shape: (num_nodes, window_size)
    
    # Adjacency matrix (correlation-based)
    corr_matrix = np.corrcoef(data_window.values.T)
    edge_index = torch.nonzero(torch.tensor(corr_matrix > 0.5, dtype=torch.bool), as_tuple=False).T

    # Edge weights (use absolute correlation values)
    edge_weights = torch.tensor(corr_matrix[edge_index[0], edge_index[1]], dtype=torch.float)

    # Create PyTorch Geometric Data object
    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights)

    return graph
