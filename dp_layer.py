# dp_layer.py
import numpy as np

def add_dp_noise(X, epsilon=1.0):
    sensitivity = 1.0
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, X.shape)
    return X + noise
