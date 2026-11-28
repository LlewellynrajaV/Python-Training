import torch

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

loss = (w * 2 + b - 10) ** 2
loss.backward()

print(f"w.grad {w.grad.item()}, b.grad {b.grad.item()}")  # w.grad -32.0, b.grad -16.0

loss_again = (w * 2 + b - 10) ** 2
loss_again.backward()
print(f"Without zeroing -> w.grad: {w.grad.item()}, b.grad: {b.grad.item()}")
# Because PyTorch accumulates gradients by default, w.grad becomes -64.0 and b.grad becomes -32.0!

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.01

for step in range(30):
  loss = (w * 2 + b - 10) ** 2
  loss.backward()

  with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad

  w.grad.zero_()
  b.grad.zero_()

  if (step + 1) in (1, 10, 30):
    print(f"Step {step + 1:2d} | loss: {loss.item():.4f} | w: {w.item():.4f}, b: {b.item():.4f}")