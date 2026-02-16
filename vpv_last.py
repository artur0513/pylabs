import numpy as np
import matplotlib.pyplot as plt
import math


x = []
y = []

N = 1000 # electrons

N1db = 1000

def E(B):
    E = 0
    for i in range(0, N, 1):
        E += (math.floor(i/N1db/B) + 0.5)*B
    return E

step = 0.0005
for B in np.linspace(1/6, 1, num=10000):
    x.append(B)
    y.append((E(B+step) - E(B-step))/2/step)

x=np.asarray(x)
x=1/x

plt.grid()
plt.plot(x, y)
plt.xlabel(r'Величина обратная к полю 1/B', fontsize=14)
plt.ylabel(r'Магнитный момент, $\mu$', fontsize=14)

plt.show()
