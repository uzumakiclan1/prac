#Hopfield Network

import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for pattern in patterns:
            pattern = pattern.reshape(self.num_neurons, 1)
            self.weights += np.dot(pattern, pattern.T)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, max_iterations=10):
        pattern = pattern.copy()
        for _ in range(max_iterations):
            for i in range(self.num_neurons):
                net_input = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if net_input >= 0 else -1
        return pattern


patterns = np.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
])

hopfield_net = HopfieldNetwork(num_neurons=patterns.shape[1])
hopfield_net.train(patterns)

test_pattern = np.array([1, -1, -1, -1])
recalled_pattern = hopfield_net.recall(test_pattern)

print("Original pattern:", test_pattern)
print("Recalled pattern:", recalled_pattern)
