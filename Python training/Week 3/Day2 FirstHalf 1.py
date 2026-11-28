import numpy as np
import torch

t_tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)
t_zeros = torch.zeros((2, 3))
t_ones = torch.ones((2, 3), dtype=torch.int64)
t_arange = torch.arange(0, 12, step=1)
t_randn = torch.randn((3, 4))

for name, t in [
    ("tensor", t_tensor),
    ("zeros", t_zeros),
    ("ones", t_ones),
    ("arange", t_arange),
    ("randn", t_randn),
]:
  print(f"{name:6s} -> shape: {t.shape}, dtype: {t.dtype}, device: {t.device}")

print("\nFirst row:", t_randn[0])
print("Second column:", t_randn[:, 1])
print("Sub-slice (rows 0..1, cols 1..2):\n", t_randn[0:2, 1:3])

reshaped_view = t_arange.view(3, 4)
reshaped_call = t_arange.reshape(3, 4)
transposed = reshaped_view.t()  
print("Reshaped shape:", reshaped_view.shape, "Transposed shape:", transposed.shape)

a = torch.randn(3, 4)
b = torch.randn(4, 2)

out_at = a @ b
out_matmul = torch.matmul(a, b)
assert torch.allclose(out_at, out_matmul)
print(f"\n(3x4) @ (4x2) -> {out_at.shape}")

np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
from_np = torch.from_numpy(np_arr)  # shares memory with np_arr
back_to_np = from_np.numpy()        # shares memory with from_np
print("From NumPy:", from_np, "| Back to NumPy type:", type(back_to_np))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
moved_tensor = out_at.to(device)
print(f"Device: {moved_tensor.device}")