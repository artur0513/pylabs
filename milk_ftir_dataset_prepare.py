import os
import numpy as np
from scipy.signal import filtfilt
import matplotlib.pyplot as plt


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


def filter_noise(y, n=5):
    if n==1:
        return y
    # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    return filtfilt(b, a, y)


def filter_spectre(x, y, n):
    return x, filter_noise(y, n)


def specter_pipeline(y_base, filename):
    x, y = read_file(filename)
    y /= y_base
    #y = filter_noise(y)
    return x, y


def get_base_y(dir):
    _, y_base0 = read_file(dir + 'окно 0.dat')
    _, y_base1 = read_file(dir + 'окно 1.dat')
    _, y_base2 = read_file(dir + 'окно 2.dat')
    return (y_base0 + y_base1 + y_base2) / 3.0


def save_to_file(x, y, filename):
    if len(x) != len(y):
        print('FUCK!!!!')
        return

    file = open(filename, 'w')
    for i in range(len(x)):
        if x[i] < 1000:
            file.write(f'{x[i]:.4f}\t{y[i]}\n')
        else:
            file.write(f'{x[i]:.3f}\t{y[i]}\n')
    file.close()

r'''
dir = r'C:\Users\artur\Desktop\MILK\28.10 ftir milk\\'
result_dir = r'C:\Users\artur\Desktop\MILK\milk ftir dataset\\'

base_y = get_base_y(dir)
for subdir, dirs, files in os.walk(dir):
    for file in files:
        path = dir + '\\' + file

        if 'обр' in path:
            x, y = specter_pipeline(base_y, path)
            save_to_file(x, y, result_dir + '\\' + file)


dir = r'C:\Users\artur\Desktop\MILK\20.09\\'

base_y = get_base_y(dir)
for subdir, dirs, files in os.walk(dir):
    for file in files:
        path = dir + '\\' + file

        if 'см1' in path:
            name, ext = file.split('.')
            _, specter_id = name.split('_')

            x, y = specter_pipeline(base_y, path)
            save_to_file(x, y, result_dir + '\\обр3 ' + specter_id + '.dat')

        if 'см2' in path:
            name, ext = file.split('.')
            _, specter_id = name.split('_')
            specter_id = str(int(specter_id) + 26)

            x, y = specter_pipeline(base_y, path)
            save_to_file(x, y, result_dir + '\\обр2 ' + specter_id + '.dat')
'''

n=6
plt.plot(*filter_spectre(*read_file(r'C:\Users\artur\Desktop\MILK\milk ftir dataset\обр0 4.dat'), n))
plt.plot(*filter_spectre(*read_file(r'C:\Users\artur\Desktop\MILK\milk ftir dataset\обр3 8.dat'), n))
plt.show()
