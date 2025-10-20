import matplotlib.pyplot as plt
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


cut_points = 200

plt.tight_layout()


def linear_mapping(x, a, b):
    return x * a + b


def x_axis_wavelength(x, y):
    maxima_indices, _ = find_peaks(y)
    x_max = [x[i] for i in maxima_indices]
    x_real = [5.0*i for i in range(len(maxima_indices))]
    print(x_max)
    est = 5*9/(0.929 - 0.095)
    popt, _ = curve_fit(linear_mapping, x_max, x_real, p0=[est, 0.0])
    a, b = popt
    print('X Transform koeff: ', a, ' (estimated by hand: ', est, ')')
    return linear_mapping(x, *popt)


def linear_base(x_arr, y_arr):
    y_lin = []

    x0 = x_arr[0]
    x1 = x_arr[len(y_arr) - 1]
    y0 = y_arr[0]
    y1 = y_arr[len(y_arr)-1]
    for x in x_arr:
        y_lin.append(y0 + (x-x0)*(y1-y0)/(x1-x0))
    return y_lin


def plot(file_path, normalize=True, label=''):
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
    y /= np.max(y)

    x = x_axis_wavelength(x, y)

    minima_indices, _ = find_peaks(-y)
    x = x[minima_indices[0]:minima_indices[1]]
    y = y[minima_indices[0]:minima_indices[1]]
    y -= linear_base(x, y)

    maxima_indices, _ = find_peaks(y)
    x -= x[maxima_indices[0]]

    plt.plot(x, y, label=label)
    #plt.plot(x, background(x, y), label='linear')


plot(r'C:\Users\artur\Desktop\MILK\02.10 крыло рэлея\specter\15 микролитров не обр 10% (убрал один из фильтров (2).txt',
     label='Раствор 15 мкл молока без обработки, P=15%')
plot(r'C:\Users\artur\Desktop\MILK\02.10 крыло рэлея\specter\дистилят 10%.txt',
     label='Дистиллят, P=10%')
plt.title('Спектр рассеяния вперед')
plt.xlabel('$см^{-1}$')
plt.legend()
plt.grid()
plt.show()
