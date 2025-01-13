import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch_geometric.data import DataLoader as GNNDataLoader
from data.dataloader import StockDataset
from models.gnn_model_with_node_events import GNNModelWithNodeEvents
from loss import CustomLoss
from torch.utils.tensorboard import SummaryWriter

def train_model():
    # TensorBoard setup
    writer = SummaryWriter("runs/gnn_with_events")

    # Initialize dataset and dataloader
    dataset = StockDataset("data/stock_data.csv", "data/events.csv", window_size=30)
    dataloader = GNNDataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model
    input_dim = 1  # Input node feature size
    hidden_dim = 16
    output_dim = 1  # Predict one value per node (e.g., tomorrow's return)
    event_dim = len(dataset.events["event_type"].unique())
    model = GNNModelWithNodeEvents(input_dim, hidden_dim, output_dim, event_dim)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CustomLoss()

    # Training loop
    for epoch in range(50):  # Train for 50 epochs
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            predictions = model(batch)
            loss = criterion(predictions, batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Log to TensorBoard
        writer.add_scalar("Loss/train", epoch_loss / len(dataloader), epoch)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")

    writer.close()

if __name__ == "__main__":
    train_model()
