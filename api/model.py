import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from pathlib import Path

MODEL_PATH = Path("notebooks/model.pt")

class AMLGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 64)
        self.conv2 = SAGEConv(64, 128)
        self.conv3 = SAGEConv(128, 64)
        self.fc    = torch.nn.Linear(64, out_channels)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x

def load_model():
    model = AMLGraphSAGE(in_channels=4, hidden_channels=128, out_channels=2)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
    return model
