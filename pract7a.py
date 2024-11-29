#Line Separation

import numpy as np
import matplotlib.pyplot as plt

def create_distance_function(a, b, c):
    def distance(x, y):
        nom = a * x + b * y + c
        if nom == 0:
            pos = 0
        elif (nom < 0 and b < 0) or (nom > 0 and b > 0):
            pos = -1
        else:
            pos = 1
        return (np.abs(nom) / np.sqrt(a ** 2 + b ** 2), pos)
    return distance

points = [(3.5, 1.8), (1.1, 3.9)]
fig, ax = plt.subplots()
ax.set_xlabel("Sweetness")
ax.set_ylabel("Sourness")
ax.set_xlim([-1, 6])
ax.set_ylim([-1, 8])

for index, (x, y) in enumerate(points):
    color = "orange" if index == 0 else "yellow"
    ax.plot(x, y, "o", color=color, markersize=10)

X = np.arange(-0.5, 5, 0.1)
step = 0.05
for x in np.arange(0, 1 + step, step):
    slope = np.tan(np.arccos(x))
    dist4line1 = create_distance_function(slope, -1, 0)
    Y = slope * X
    results = [dist4line1(*point) for point in points]
    if results[0][1] != results[1][1]:
        ax.plot(X, Y, "g-", label="Decision Boundary")
    else:
        ax.plot(X, Y, "r-", label="Decision Boundary")

plt.show()
