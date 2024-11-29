#Hopfield associative memory Network model 

import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, n):
        self.n = n
        self.weights = np.zeros((n, n))
        self.neurons = np.zeros(n)

    def train(self, patterns):
        num_patterns = len(patterns)
        for p in patterns:
            p = 2 * p - 1
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)

    def recall(self, input_pattern, steps=5):
        input_pattern = 2 * input_pattern - 1
        self.neurons = input_pattern
        for _ in range(steps):
            for i in range(self.n):
                summation = np.dot(self.weights[i], self.neurons)
                self.neurons[i] = 1 if summation > 0 else -1
        return (self.neurons + 1) // 2

pattern1 = np.array([1, 1, 1])
pattern2 = np.array([1, 0, 1])
hopfield_net = HopfieldNetwork(3)
hopfield_net.train([pattern1, pattern2])
noisy_input = np.array([1, 0, 1])
recalled_output = hopfield_net.recall(noisy_input)
print("Noisy Input Pattern: ", noisy_input)
print("Recalled Pattern: ", recalled_output)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Pattern 2")
plt.imshow(pattern2.reshape(1, 3), cmap='binary', aspect='auto')
plt.subplot(1, 2, 2)
plt.title("Recalled Pattern")
plt.imshow(recalled_output.reshape(1, 3), cmap='binary', aspect='auto')
plt.show()
