import torch
from torch.utils.data import Dataset
import pandas as pd
from create_graph import create_graph  # Assumes you have the `create_graph` utility.

class StockDataset(Dataset):
    def __init__(self, stock_csv, event_csv, window_size=30):
        super().__init__()
        self.data = pd.read_csv(stock_csv, index_col=0)
        self.events = pd.read_csv(event_csv, index_col=0)
        self.window_size = window_size
        self.unique_nodes = self.data.columns.tolist()

    def get_event_features(self, date_range):
        """
        Fetch event features for each company in the graph during the date range.
        """
        event_data = self.events[self.events["date"].isin(date_range)]
        node_event_features = torch.zeros((len(self.unique_nodes), len(event_data["event_type"].unique())))

        for _, event in event_data.iterrows():
            affected_node = event["affected_node"]
            if affected_node == "ALL":  # Market-wide event
                node_indices = range(len(self.unique_nodes))
            else:
                node_indices = [self.unique_nodes.index(affected_node)]

            event_type_idx = list(event_data["event_type"].unique()).index(event["event_type"])
            for node_idx in node_indices:
                node_event_features[node_idx, event_type_idx] += event["event_strength"]

        return node_event_features

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        data_window = self.data.iloc[idx:idx + self.window_size]
        date_range = self.data.index[idx:idx + self.window_size]
        
        graph = create_graph(data_window)
        
        # Add event features
        event_features = self.get_event_features(date_range)
        graph.event_features = event_features

        # Target: Predict next day’s returns
        target_returns = self.data.iloc[idx + self.window_size].pct_change().values
        graph.y = torch.tensor(target_returns, dtype=torch.float)

        return graph
