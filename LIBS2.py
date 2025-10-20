import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import lfilter, filtfilt
from scipy.optimize import curve_fit
import os
import re
from matplotlib.widgets import Cursor


matplotlib.rcParams['savefig.dpi'] = 300
fig, ax = plt.subplots(figsize=(12.8, 1840.0/300.0))
fig.tight_layout(pad=0)
plt.xlabel('нм')

normalize = True
smooth = True
wm = plt.get_current_fig_manager()
wm.window.state('zoomed')

def draw_vertical_line(x: float, text: str = "", up: bool = True):
    y_max = 2.5
    y_min = 0
    y_arr = np.array([y_max, y_min]) * 1.1
    plt.plot([x, x], y_arr, color='lightgray', zorder=-10, linestyle='dashed')

    if len(text) > 0:
        if up:
            plt.text(x - 3, np.max(y_arr), text)
        else:
            plt.text(x - 3, np.min(y_arr), text, verticalalignment='top')


def read_file(filename: str) -> [np.array, np.array]:
    file = open(filename)
    x = []
    y = []

    # Read file
    for line in file:
        if line[0].isnumeric() or line[0] == '-':
            values = line.split('\t')
            if float(values[0].replace(',', '.')) > 885:
                continue
            x.append(float(values[0].replace(',', '.')))
            y.append(float(values[1].replace(',', '.')))

    return np.array(x), np.array(y)


def avg_data_from_folder(folder: str):
    x = []
    y_list = []

    for subdir, dirs, files in os.walk(folder):
        for file in files:
            path = folder + '\\' + file
            x, y = read_file(path)
            if normalize:
                y /= np.median(y)
            if smooth:
                n = 6  # the larger n is, the smoother curve will be
                b = [1.0 / n] * n
                a = 1
                y = filtfilt(b, a, y)
            y_list.append(y)

    axis = 0
    y_avg = np.mean(y_list, axis=axis)
    y_dev = np.std(y_list, axis=axis)
    return x, y_avg, y_dev


color_counter = 0


def plot_avg_data_from_folder(folder: str, label=''):
    global color_counter
    x, y_avg, y_dev = avg_data_from_folder(folder)
    plt.plot(x, y_avg, label=label)
    color = 'lightblue'
    if color_counter == 1:
        color = 'peachpuff'
    if color_counter == 2:
        color = 'lightgreen'
    if color_counter == 2:
        color = 'lightcoral'
    plt.fill_between(x, y_avg - y_dev, y_avg + y_dev, color=color, zorder=-10, alpha=0.33)
    #plt.plot(x, y_avg - y_dev, color='gray')
    #plt.plot(x, y_avg + y_dev, color='gray')
    color_counter += 1


# Important
def draw_lines(all_lines=False):
    draw_vertical_line(766.49, "K")
    draw_vertical_line(769.90)

    draw_vertical_line(589.00, "Na")

    draw_vertical_line(616.17, "Ca\nNa\nP(II)", False)
    draw_vertical_line(612.07, "Ca\nK(II)")
    draw_vertical_line(526.77, "Ca")
    draw_vertical_line(558.87, "Ca")

    draw_vertical_line(393.33, "Ca")
    draw_vertical_line(396.83)

    if all_lines:
        draw_vertical_line(247.88, "C", False)

        draw_vertical_line(253.57, "P")
        draw_vertical_line(255.28)

        draw_vertical_line(279.55, "Mg", False)
        draw_vertical_line(285.21)

        draw_vertical_line(422.67, "Ca", False)
        draw_vertical_line(643.90, "Ca")
        draw_vertical_line(646.90)

        draw_vertical_line(656.26, "H")
        draw_vertical_line(434.05, "H")

        draw_vertical_line(777.35, "O")
        draw_vertical_line(844.67, "O")

        draw_vertical_line(746.89, "N")
        draw_vertical_line(821.82, "N")
        draw_vertical_line(868.28, "N")

        draw_vertical_line(500.36, "N(II)")
        draw_vertical_line(399.48, "N(II)", False)
        draw_vertical_line(463.05, "N(II)", False)
        draw_vertical_line(567.93, "N(II)", False)
        draw_vertical_line(332.96, "N(II)", False)
        draw_vertical_line(343.71)


#plt.xlim(510, 670)
#plt.ylim(0.8, 2.2)

#plt.xlim(740, 800)
#plt.ylim(0.6, 2.2)

#plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\1', 'см2')
#plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\2', 'в1')
#plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\3', 'в2')
#plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\4', 'см1')
#plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\5', 'н')
#plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\6', 'м88')
#plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\7', 'м98')
#plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\8', 'м76')


scenario = 4
folder = r'C:\Users\artur\Desktop\MILK\libs_avg_plots\final\\'

if scenario == 0:
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\2', 'в1')
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\6', 'м88')
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\7', 'м98')
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\8', 'м76')
    draw_lines()
    plt.legend()
    plt.savefig(folder+'в1_м76_м88_м95.png')

    plt.xlim(510, 670)
    plt.ylim(0.8, 2.2)
    plt.savefig(folder+'в1_м76_м88_м95 (500-660 нм).png')

    plt.xlim(740, 800)
    plt.ylim(0.6, 2.4)
    plt.savefig(folder+'в1_м76_м88_м95 (700-800 нм).png')
elif scenario == 1:
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\3', 'в2')
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\6', 'м88')
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\7', 'м98')
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\8', 'м76')
    draw_lines()
    plt.legend()
    plt.savefig(folder+'в2_м76_м88_м95.png')

    plt.xlim(510, 670)
    plt.ylim(0.8, 2.2)
    plt.savefig(folder+'в2_м76_м88_м95 (500-660 нм).png')

    plt.xlim(740, 800)
    plt.ylim(0.6, 2.4)
    plt.savefig(folder+'в2_м76_м88_м95 (700-800 нм).png')
elif scenario == 2:
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\5', 'н')
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\2', 'в1')
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\3', 'в2')
    draw_lines()
    plt.legend()
    plt.savefig(folder+'н_в1_в2.png')

    plt.xlim(510, 670)
    plt.ylim(0.8, 2.2)
    plt.savefig(folder+'н_в1_в2 (500-660 нм).png')

    plt.xlim(740, 800)
    plt.ylim(0.6, 2.4)
    plt.savefig(folder+'н_в1_в2 (700-800 нм).png')
elif scenario == 3:
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\5', 'н')
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\4', 'см1')
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\1', 'см2')
    draw_lines()
    plt.legend()
    plt.savefig(folder+'н_см1_см2.png')

    plt.xlim(510, 670)
    plt.ylim(0.8, 2.2)
    plt.savefig(folder+'н_см1_см2 (500-660 нм).png')

    plt.xlim(740, 800)
    plt.ylim(0.6, 2.4)
    plt.savefig(folder+'н_см1_см2 (700-800 нм).png')
elif scenario == 4:
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\5', 'н')
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\6', 'м88')
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\7', 'м98')
    plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\8', 'м76')
    draw_lines()
    plt.legend()
    plt.savefig(folder + 'н_м76_м88_м95.png')
    plt.savefig(folder + 'н_м76_м88_м95.svg')

    plt.xlim(510, 670)
    plt.ylim(0.8, 2.2)
    plt.savefig(folder + 'н_м76_м88_м95 (500-660 нм).png')

    plt.xlim(740, 800)
    plt.ylim(0.6, 2.4)
    plt.savefig(folder + 'н_м76_м88_м95 (700-800 нм).png')


draw_lines(True)
#plot_avg_data_from_folder(r'C:\Users\artur\Desktop\MILK\13.10 libs milk\2', 'peachpuff')
#cursor = Cursor(ax, horizOn=False, vertOn=True, useblit=True, color='black', linewidth=1)
# plt.grid()
# plt.legend()
plt.show()
