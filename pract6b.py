#Adaptive resonance theory

import numpy as np

class ART1:
    def __init__(self, num_features, num_categories, vigilance=0.75):
        self.vigilance = vigilance
        self.weights = np.ones((num_categories, num_features))

    def train(self, data):
        for sample in data:
            activations = np.dot(self.weights, sample)
            category = np.argmax(activations)
            if self._vigilance_test(sample, category):
                self.weights[category] = np.minimum(self.weights[category], sample)

    def _vigilance_test(self, sample, category):
        return np.dot(self.weights[category], sample) / np.sum(sample) >= self.vigilance

    def predict(self, sample):
        activations = np.dot(self.weights, sample)
        category = np.argmax(activations)
        return category if self._vigilance_test(sample, category) else None

data = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]])
art1 = ART1(num_features=data.shape[1], num_categories=3, vigilance=0.75)
art1.train(data)

test_sample = np.array([1, 0, 1, 0])
print("Data Belongs to cat :", art1.predict(test_sample))

