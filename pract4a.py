#Write a program for Back Propagation Algorithm

import numpy as np
import math

np.set_printoptions(precision=2)

v1 = np.array([0.6, 0.3])
v2 = np.array([-0.1, 0.4])
w = np.array([-0.2, 0.4, 0.1])
b1 = 0.3
b2 = 0.5
x1 = 0
x2 = 1
alpha = 0.25

zin1 = round(b1 + x1 * v1[0] + x2 * v2[0], 4)
print("Net input to z1 layer (zin1):", round(zin1, 3))
zin2 = round(b2 + x1 * v1[1] + x2 * v2[1], 4)
print("Net input to z2 layer (zin2):", round(zin2, 4))

z1 = 1 / (1 + math.exp(-zin1))
z1 = round(z1, 4)
z2 = 1 / (1 + math.exp(-zin2))
z2 = round(z2, 4)
print("Output z1:", z1)
print("Output z2:", z2)

yin = w[0] + z1 * w[1] + z2 * w[2]
print("Net input to output layer (yin):", yin)

y = 1 / (1 + math.exp(-yin))
print("Output y:", y)

fyin = y * (1 - y)
dk = (1 - y) * fyin
print("dk:", dk)

dw1 = alpha * dk * z1
dw2 = alpha * dk * z2
dw0 = alpha * dk
print("Weight changes for output layer:")
print("dw1:", dw1)
print("dw2:", dw2)
print("dw0:", dw0)

din1 = dk * w[1]
din2 = dk * w[2]
print("Error in delta for hidden layer:")
print("din1:", din1)
print("din2:", din2)

fzin1 = z1 * (1 - z1)
fzin2 = z2 * (1 - z2)
d1 = din1 * fzin1
d2 = din2 * fzin2
print("d1:", d1)
print("d2:", d2)

dv11 = alpha * d1 * x1
dv21 = alpha * d1 * x2
dv01 = alpha * d1
dv12 = alpha * d2 * x1
dv22 = alpha * d2 * x2
dv02 = alpha * d2
print("Weight changes for hidden layer:")
print("dv11:", dv11)
print("dv21:", dv21)
print("dv01:", dv01)
print("dv12:", dv12)
print("dv22:", dv22)
print("dv02:", dv02)

v1[0] = v1[0] + dv11
v1[1] = v1[1] + dv12
v2[0] = v2[0] + dv21
v2[1] = v2[1] + dv22
w[1] = w[1] + dw1
w[2] = w[2] + dw2
w[0] = w[0] + dw0
b1 = b1 + dv01
b2 = b2 + dv02

print("Final weights and biases:")
print("Updated weights v1:", v1)
print("Updated weights v2:", v2)
print("Updated weights w:", w)
print("Updated biases b1:", b1, " b2:", b2)
