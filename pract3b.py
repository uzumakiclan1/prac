#Write a program to implement of delta rule

import numpy as np

x = np.array([float(input("Initial input: ")) for _ in range(3)])
weights = np.array([float(input("Initial weight: ")) for _ in range(3)])
desired = np.array([float(input("Desired output: ")) for _ in range(3)])
a = float(input("Enter learning rate: "))

actual = x * weights
print("Initial actual output:", actual)

while not np.allclose(desired, actual):
    weights += a * (desired - actual) / x
    actual = x * weights
    print("Updated weights:", weights)
    print("Updated actual output:", actual)
    print("*" * 30)

print("Final output")
print("Corrected weights:", weights)
print("Final actual output:", actual)
