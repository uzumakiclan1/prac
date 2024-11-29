#Write a program to implement Hebb’s rule.

import numpy as np

x1 = np.array([1, 1, 1, -1, 1, -1, 1, 1, 1])
x2 = np.array([1, 1, 1, 1, -1, 1, 1, 1, 1])
y = np.array([1, -1])
wtold = np.zeros(9)
wtnew = np.zeros(9)
b = 0

print("First input with target = 1")
for i in range(9):
    wtold[i] = wtold[i] + x1[i] * y[0]
wtnew = wtold
b = b + y[0]
print("Updated weights:", wtnew)
print("Bias value:", b)

print("\nSecond input with target = -1")
for i in range(9):
    wtnew[i] = wtold[i] + x2[i] * y[1]
b = b + y[1]
print("Updated weights:", wtnew)
print("Bias value:", b)

