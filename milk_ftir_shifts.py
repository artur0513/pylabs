import os
import numpy as np
from scipy.signal import filtfilt
import matplotlib.pyplot as plt

dir = r'C:\Users\artur\Desktop\28.10 ftir milk\\'


def read_file(filename: str) -> [np.array, np.array]:
    file = open(filename)
    x = []
    y = []

    # Read file
    for line in file:
        if line[0].isnumeric() or line[0] == '-':
            values = line.split('\t')
            x.append(float(values[0]))
            y.append(float(values[1]))

    return np.array(x), np.array(y)


def filter_noise(y):
    n = 4  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    return filtfilt(b, a, y)


def area_under_curve(y: np.array):
    return y.sum()


def specter_pipeline(y_base, filename):
    x, y = read_file(filename)
    y /= y_base
    y = filter_noise(y)
    y = -np.log(y)

    # y /= np.max(y)
    #y /= np.quantile(y, 0.9)
    y = (y - np.mean(y)) / np.std(y)
    return x, y


def get_base_y():
    _, y_base0 = read_file(dir + 'окно 0.dat')
    _, y_base1 = read_file(dir + 'окно 1.dat')
    _, y_base2 = read_file(dir + 'окно 2.dat')
    return (y_base0 + y_base1 + y_base2) / 3.0


base_y = get_base_y()
for subdir, dirs, files in os.walk(dir):
    for file in files:
        path = dir + '\\' + file
        # if 'обр0' in path:
        if 'dat' in path and not 'окно' in path and not 'воздух' in path:
            color='black'
            if 'обр0' in path:
                color='greenyellow'
            if 'обр1' in path:
                color = 'lightcoral'
            if 'обр2' in path:
                color = 'lightsteelblue'
            plt.plot(*specter_pipeline(base_y, path), color=color, alpha=0.33)


plt.show()
