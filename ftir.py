import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, filtfilt
from scipy.optimize import curve_fit
import os
import re
from matplotlib.widgets import Cursor

fig, ax = plt.subplots(figsize=(8, 6))


def read_file(filename: str) -> [np.array, np.array]:
    file = open(filename)
    x = []
    y = []

    # Read file
    for line in file:
        if line[0].isnumeric() or line[0] == '-':
            values = line.split('\t')
            if 920 < float(values[0]) < 1900:
                x.append(float(values[0]))
                y.append(float(values[1]))

    print('Num of points: ', len(x))
    return np.array(x), np.array(y)


def filter_noise(y):
    n = 10  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    return filtfilt(b, a, y)


def process_pipeline(filename: str):
    x, y = read_file(filename)
    basename_without_ext = os.path.splitext(os.path.basename(filename))[0]
    #y = filter_noise(y)
    y = (y - np.mean(y)) / np.std(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    plt.plot(x, y, label=basename_without_ext)



#process_pipeline(r'C:\Users\artur\Desktop\MILK\ftir 08.10\N_1.dat')
#process_pipeline(r'C:\Users\artur\Desktop\MILK\ftir 08.10\76_1.dat')
#process_pipeline(r'C:\Users\artur\Desktop\MILK\ftir 08.10\88_1.dat')
#process_pipeline(r'C:\Users\artur\Desktop\MILK\ftir 08.10\98_1.dat')

process_pipeline(r'C:\Users\artur\Desktop\MILK\ftir 08.10\B1.dat')
process_pipeline(r'C:\Users\artur\Desktop\MILK\ftir 08.10\B2.dat')
process_pipeline(r'C:\Users\artur\Desktop\MILK\ftir 08.10\CM1.dat')
process_pipeline(r'C:\Users\artur\Desktop\MILK\ftir 08.10\CM2_2.dat')

plt.grid()
plt.legend()
plt.show()