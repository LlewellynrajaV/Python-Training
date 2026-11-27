import numpy as np

np.random.seed(1)

W1 = np.random.randn(3, 4)   
b1 = np.zeros(4)             
W2 = np.random.randn(4, 1)   
b2 = np.zeros(1)             

print("Shapes:")
print(f"W1: {W1.shape[0]}x{W1.shape[1]} {W1.shape}")
print(f"b1: {b1.shape[0]} {b1.shape}")
print(f"W2: {W2.shape[0]}x{W2.shape[1]} {W2.shape}")
print(f"b2: {b2.shape[0]} {b2.shape}")

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def forward(x):
    x = np.asarray(x, dtype=float)
    z1 = x @ W1 + b1   
    h = relu(z1)          
    z2 = h @ W2 + b2      
    y_hat = sigmoid(z2)   
    return h, y_hat

x_input = np.array([0.5, 1.2, 3.0])
hidden_vals, output_prob = forward(x_input)

print("\nInput:", x_input)
print("Hidden values (ReLU):", hidden_vals)
print("Output probability (Sigmoid):", output_prob)