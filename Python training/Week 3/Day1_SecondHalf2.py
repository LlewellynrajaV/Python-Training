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
val_ds = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)


class PlainMLP(nn.Module):

  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

  def forward(self, x):
    return self.net(x)


class RegularizedMLP(nn.Module):

  def __init__(self, dropout_p=0.3):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(2, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(dropout_p),
        nn.Linear(32, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(dropout_p),
        nn.Linear(32, 1),
    )

  def forward(self, x):
    return self.net(x)


def evaluate_val_loss(model, loader, criterion):
  model.eval()
  total_loss = 0.0
  with torch.no_grad():
    for bx, by in loader:
      bx, by = bx.to(device), by.to(device)
      preds = model(bx)
      total_loss += criterion(preds, by).item() * bx.size(0)
  return total_loss / len(loader.dataset)


def train_with_early_stopping(
    model, optimizer, epochs=100, patience=10, track_val=False
):
  model = model.to(device)
  criterion = nn.BCEWithLogitsLoss()

  train_losses = []
  val_losses = []
  best_val_loss = float("inf")
  epochs_no_improve = 0
  best_weights = None

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

    train_losses.append(running_loss / len(train_loader.dataset))

    if track_val:
      val_loss = evaluate_val_loss(model, val_loader, criterion)
      val_losses.append(val_loss)

      if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
      else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
          print(f"Early stopping at epoch {epoch + 1}")
          break

  if best_weights is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

  return train_losses, val_losses


sgd_model = PlainMLP()
sgd_train, _ = train_with_early_stopping(
    sgd_model, torch.optim.SGD(sgd_model.parameters(), lr=0.05), epochs=100
)

mom_model = PlainMLP()
mom_train, _ = train_with_early_stopping(
    mom_model,
    torch.optim.SGD(mom_model.parameters(), lr=0.05, momentum=0.9),
    epochs=100,
)

adam_model = PlainMLP()
adam_train, _ = train_with_early_stopping(
    adam_model, torch.optim.Adam(adam_model.parameters(), lr=0.01), epochs=100
)

plain_val_model = PlainMLP()
_, plain_val = train_with_early_stopping(
    plain_val_model,
    torch.optim.AdamW(plain_val_model.parameters(), lr=0.01, weight_decay=1e-4),
    epochs=100,
    patience=10,
    track_val=True,
)

reg_val_model = RegularizedMLP(dropout_p=0.3)
_, reg_val = train_with_early_stopping(
    reg_val_model,
    torch.optim.AdamW(reg_val_model.parameters(), lr=0.01, weight_decay=1e-4),
    epochs=100,
    patience=10,
    track_val=True,
)

print(f"Final Val Loss (Plain MLP): {plain_val[-1]:.4f}")
print(f"Final Val Loss (BatchNorm1d + Dropout 0.3): {reg_val[-1]:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(sgd_train, label="SGD")
ax1.plot(mom_train, label="SGD + Momentum")
ax1.plot(adam_train, label="Adam")
ax1.set_title("Optimiser Training Loss (100 Epochs)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(plain_val, label="Plain MLP Val Loss")
ax2.plot(reg_val, label="BatchNorm1d + Dropout(0.3) Val Loss")
ax2.set_title("Validation Loss with Early Stopping")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("task2_loss_curves.png", dpi=150)
plt.show()