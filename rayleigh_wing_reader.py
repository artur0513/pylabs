import matplotlib.pyplot as plt
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


cut_points = 150


def mapping(values_x, a, b, c):
    x = np.array(values_x)
    return a*np.exp(-b*x)+c


def linear_mapping(x, a, b):
    return x * a + b


def x_axis_wavelength(x, y):
    maxima_indices, _ = find_peaks(y)
    x_max = [x[i] for i in maxima_indices]
    x_real = [5.0*i for i in range(len(maxima_indices))]

    # Убрать 1 и ластовую точку if в спискe оHи

    print(x_max)
    est = 5*9/(0.929 - 0.095)
    popt, _ = curve_fit(linear_mapping, x_max, x_real, p0=[est, 0.0])
    a, b = popt
    print('X Transform koeff: ', a, ' (estimated by hand: ', est, ')')
    return linear_mapping(x, *popt)


def background(x, y):
    minima_indices, _ = find_peaks(-y)
    y_mins = [y[i] for i in minima_indices]
    x_mins = [x[i] for i in minima_indices]
    popt, _ = curve_fit(mapping, x_mins, y_mins, p0=[0.15, 2.5, 0.02])
    print(*popt)
    y_fit1 = mapping(x, *popt)
    return y_fit1


def plot(file_path, normalize=True):
    basename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
    file = open(file_path, 'r')

    x = []
    y = []

    for line in file:
        if line[0] != '#':
            x.append(float(line.split(" ")[0]))
            y.append(float(line.split(" ")[1]))
    x = np.array(x[cut_points:])
    y = np.array(y[cut_points:])
    y -= background(x, y)
    y /= np.max(y)

    x = x_axis_wavelength(x, y)

    #x /= (0.929 - 0.095)/9
    #x *= 5
    plt.plot(x, y, label=basename_without_ext)
    #plt.plot(x, background(x, y), label='linear')


root = tk.Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(initialdir=r'C:\Users\artur\Desktop\MILK')
for path in file_paths:
    plot(path)

plt.legend()
plt.show()
