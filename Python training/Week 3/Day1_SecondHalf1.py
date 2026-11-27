import matplotlib.pyplot as plt


def run_gd(lr, steps=50, print_every=10, label=None):
  x, t = 2.0, 10.0
  w, b = 1.0, 0.0
  losses = []

  for step in range(steps):
    pred = w * x + b
    error = pred - t
    loss = error**2
    losses.append(loss)

    if label and step % print_every == 0:
      print(
          f"[{label}] Step {step:2d} | Loss: {loss:.6f} | w: {w:.4f} | b:"
          f" {b:.4f}"
      )

    dw = 2.0 * error * x
    db = 2.0 * error

    w -= lr * dw
    b -= lr * db

  return losses


print("--- lr = 0.05 (50 steps) ---")
losses_005 = run_gd(lr=0.05, steps=50, print_every=10, label="lr=0.05")

losses_0001 = run_gd(lr=0.001, steps=50)
losses_05 = run_gd(lr=0.5, steps=50)

plt.figure(figsize=(8, 5))
plt.plot(losses_0001, label="lr = 0.001 (slow convergence)")
plt.plot(losses_005, label="lr = 0.05 (steady convergence)")
plt.plot(losses_05, label="lr = 0.5 (diverges)")
plt.yscale("log")  # Log scale makes divergence and small losses easy to see
plt.title("Gradient Descent Loss Curves Across Learning Rates")
plt.xlabel("Step")
plt.ylabel("Loss (log scale)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("task1_loss_curves.png", dpi=150)
plt.show()