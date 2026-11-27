import numpy as np
import matplotlib.pyplot as plt
def relu(x):
    return np.maximum(0, x)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    # Subtract max for numerical stability
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)

probs = softmax([2.0, 1.0, 0.1])
print("Softmax([2.0, 1.0, 0.1]):", np.round(probs, 3))
print("Sum of softmax outputs:", np.sum(probs))


def tanh_fn(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, alpha * x)

x_vals = np.linspace(-6, 6, 400)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, relu(x_vals), label="ReLU", linewidth=2)
plt.plot(x_vals, sigmoid(x_vals), label="Sigmoid", linewidth=2)
plt.plot(x_vals, tanh_fn(x_vals), label="Tanh", linewidth=2)
plt.title("Activation Functions (-6 to 6)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
plt.axvline(0, color="gray", linewidth=0.8, linestyle="--")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy(y_true, y_prob, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    # Binary cross-entropy for probabilities in [0, 1]
    return -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))

y_true_3 = np.array([1.0, 0.0, 1.0])
y_pred_3 = np.array([0.8, 0.2, 0.7])
print("\nMSE (3 predictions):", mse(y_true_3, y_pred_3))
print("Cross-Entropy (3 predictions):", cross_entropy(y_true_3, y_pred_3))

confident_wrong_ce = cross_entropy(np.array([1.0]), np.array([0.01]))
uncertain_ce = cross_entropy(np.array([1.0]), np.array([0.50]))

print(f"\nTrue label = 1 | Confident wrong (p=0.01) CE: {confident_wrong_ce:.4f}")
print(f"True label = 1 | Uncertain (p=0.50) CE:      {uncertain_ce:.4f}")