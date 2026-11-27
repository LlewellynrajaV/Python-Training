import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)


class MLP(nn.Module):

  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(2, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 1),
    )

  def forward(self, x):
    return self.net(x)


model = MLP().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

epochs = 100
train_losses = []
lr_over_time = []

for epoch in range(epochs):
  model.train()
  running_loss = 0.0
  for bx, by in train_loader:
    bx, by = bx.to(device), by.to(device)
    optimizer.zero_grad()
    loss = criterion(model(bx), by)
    loss.backward()
    optimizer.step()
    running_loss += loss.item() * bx.size(0)

  train_losses.append(running_loss / len(train_ds))
  lr_over_time.append(optimizer.param_groups[0]["lr"])
  scheduler.step()

plt.figure(figsize=(7, 4.5))
plt.plot(range(1, epochs + 1), lr_over_time, marker="o", markersize=3)
plt.title("StepLR Learning Rate Over Time (step_size=25, gamma=0.5)")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("bonus_steplr_schedule.png", dpi=150)
plt.show()