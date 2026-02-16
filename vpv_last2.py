import time

import numpy as np
import matplotlib.pyplot as plt
import math

x = []
y = []

N = 1000  # electrons
kT = 0.06

N1db = 1000


def E(B):
    E = 0

    Ef_start = (math.ceil(N / N1db / B) + 0.5) * B

    def Ef_find(Ef):
        numtotal = 0
        for i in range(0, math.ceil(N / N1db / B) + 5, 1):
            E1 = (i + 0.5) * B
            num = N1db * B / (np.exp((E1 - Ef) / kT) + 1)
            numtotal += num
        return numtotal

    Ef_true = Ef_start
    n = 1

    a = Ef_start * 2.0
    b = 0

    if (Ef_find(a) - N) * (Ef_find(b) - N) > 0:
        print('FFFUUUUUCCCCKKKKK!!!!')
        exit()

    tstart = round(time.time() * 1000)
    while abs(Ef_find(Ef_true) - N) > 0.01:
        n += 1

        Ef_true = (a + b) / 2
        if (Ef_find(a) - N) * (Ef_find(Ef_true) - N) < 0:
            b = Ef_true
        else:
            a = Ef_true

    print('Ef_true found for B=', B, round(time.time() * 1000) - tstart, '  iterations: ', n)
    numtotal = 0
    for i in range(0, math.ceil(N / N1db / B) + 5, 1):
        E1 = (i + 0.5) * B
        num = N1db * B / (np.exp((E1 - Ef_true) / kT) + 1)
        E += E1 * num
        numtotal += num

    # print(numtotal)
    return E - Ef_start*(numtotal - N)


for B in np.linspace(1 / 10, 1, num=1000):
    step = B / 20
    x.append(B)
    y.append((E(B - 2 * step) / 12 - 2 / 3 * E(B - step) + 2 / 3 * E(B + step) - E(B + 2 * step) / 12) / step)

x = np.asarray(x)
y = np.asarray(y)
y = y / x
x = 1 / x

plt.grid()
plt.plot(x, y)
plt.xlabel(r'Величина обратная к полю 1/B', fontsize=14)
plt.ylabel(r'Магнитный момент, $\mu / B$', fontsize=14)

plt.show()
