import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))    # (N, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))    # (N, 64,  7,  7)
        x = torch.flatten(x, 1)                 # (N, 3136)
        x = self.dropout(F.relu(self.fc1(x)))   # (N, 128)
        return self.fc2(x)                      # (N, 10)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SimpleCNN()
    out = model(torch.randn(4, 1, 28, 28))
    print(f"output shape: {tuple(out.shape)}")
    print(f"trainable parameters: {count_parameters(model):,}")
