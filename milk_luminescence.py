import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from scipy.special import erfc
from scipy.optimize import curve_fit


def read_luminescence_txt(filename: str, id: int = 0) -> [np.array, np.array]:
    file = open(filename, encoding='Windows-1251')
    nm = []
    intensity = []

    data_start_counter = 0
    for line in file:
        if 'nm\tData' in line:
            data_start_counter += 1
            continue

        if data_start_counter == id + 1:
            if not (line[0].isnumeric() or line[0] == '-'):
                continue
            values = line.split('\t')
            nm_value = float(values[0])
            nm.append(nm_value)
            intensity.append(float(values[1]))

            if nm_value > 599.9:
                break

    return np.array(nm), np.array(intensity)


def filter_noise(y, n=6):
    b = [1.0 / n] * n
    a = 1
    return filtfilt(b, a, y)


def plot_luminescence_txt(filename: str):
    for id in range(5):
        nm, intensity = read_luminescence_txt(filename, id)
        intensity = filter_noise(intensity, 6)
        file_name_with_ext = Path(filename).name

        # plt.figure(figsize=(12.8, 9.6))
        plt.plot(nm, intensity, label=file_name_with_ext + f" (id: {id})")

    plt.ylabel("Сигнал, усл.ед.")
    plt.xlabel("Длина волны, нм")
    plt.grid()
    plt.tight_layout()
    # plt.show()

plt.figure(figsize=(12.8, 9.6))
base_dir = "C:\\Users\\artur\\Desktop\\Milk PCA feb 26\\люминесценция\\"

base_dir_new = "C:\\Users\\artur\\Desktop\\Milk PCA feb 26\\люминесценция новая\\"

# plot_luminescence_txt(base_dir_new + "образец 2\\2_ex_290_1.TXT")
# plot_luminescence_txt(base_dir_new + "образец 2\\2_ex_290_2.TXT")
# plot_luminescence_txt(base_dir_new + "образец 2\\2_ex_290_3.TXT")
# plot_luminescence_txt(base_dir_new + "образец 2\\2_ex_290_4.TXT")

plot_luminescence_txt(base_dir_new + "образец 2\\2_ex_360_1.TXT")
plot_luminescence_txt(base_dir_new + "образец 2\\2_ex_360_2.TXT")
plot_luminescence_txt(base_dir_new + "образец 2\\2_ex_360_3.TXT")
plot_luminescence_txt(base_dir_new + "образец 2\\2_ex_360_4.TXT")


'''
plot_luminescence_txt(base_dir + "1_исходный_290_1.TXT")
plot_luminescence_txt(base_dir + "1_исходный_290_2.TXT")
plot_luminescence_txt(base_dir + "1_исходный_360_1.TXT")
plot_luminescence_txt(base_dir + "1_исходный_360_2.TXT")
plot_luminescence_txt(base_dir + "1_разбавленный_290_1.TXT")
plot_luminescence_txt(base_dir + "1_разбавленный_290_2.TXT")
plot_luminescence_txt(base_dir + "1_разбавленный_360_1.TXT")
plot_luminescence_txt(base_dir + "1_разбавленный_360_2.TXT")
'''

plt.legend()
plt.show()