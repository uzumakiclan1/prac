#Generate AND/NOT function using McCulloch-Pitts neural net.

num = int(input("Enter number of inputs: "))

w1 = 1
w2 = 1
x1 = []
x2 = []

for i in range(num):
    ele1 = int(input("x1= "))
    ele2 = int(input("x2= "))
    x1.append(ele1)
    x2.append(ele2)

n = [x1[i] * w1 for i in range(num)]
m = [x2[i] * w2 for i in range(num)]

yin = []
for i in range(num):
    yin.append(n[i] + m[i])
print("Y-IN:", yin)

yin_inhibitory = []
for i in range(num):
    yin_inhibitory.append(n[i] - m[i])
print("After assuming one weight as excitatory and the other as inhibitory Y-IN:", yin_inhibitory)

Y = []
for i in range(num):
    if yin_inhibitory[i] >= 1:
        ele = 1
    else:
        ele = 0
    Y.append(ele)
print("Y:", Y)
