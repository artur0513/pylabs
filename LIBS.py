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
            if float(values[0].replace(',', '.')) > 885:
                continue
            x.append(float(values[0].replace(',', '.')))
            y.append(float(values[1].replace(',', '.')))

    return np.array(x), np.array(y)


class Line:
    x: float = 0.0
    text: str = ""


class FilePlotter(object):
    y_min = 2 ** 31
    y_max = -2 ** 31
    line_counter = 0

    def plot(self, filename: str, y_shift: float = 0.0, y_mul: float = 1.0, color=""):
        x, y = read_file(filename)
        new_y = y * y_mul + y_shift
        self.y_min = min(self.y_min, np.min(new_y))
        self.y_max = max(self.y_max, np.max(new_y))
        plt.plot(x, new_y)

    def draw_vertical_line(self, x: float, text: str = "", up: bool = True):
        y_arr = np.array([self.y_max, self.y_min]) * 1.1
        plt.plot([x, x], y_arr, color='lightgray', zorder=-10, linestyle='dashed')

        if len(text) > 0:
            if up:
                plt.text(x - 3, np.max(y_arr), text)
            else:
                plt.text(x - 3, np.min(y_arr) * 1.05, text)
        self.line_counter += 1


plotter = FilePlotter()

plotter.plot(r'C:\Users\artur\Desktop\30.09 milk\без обр\15%_Subt2__0__14-00-51-150.txt')
plotter.plot(r'C:\Users\artur\Desktop\30.09 milk\95 градусов\15%_Subt2__1__15-10-10-186.txt')

plotter.plot(r'C:\Users\artur\Desktop\30.09 milk\высушенный без обр\15%_Subt2__3__15-19-13-280.txt', -10000.0)
plotter.plot(r'C:\Users\artur\Desktop\30.09 milk\высушенный 95 градусов\15%_Subt2__3__16-12-47-419.txt', -10000.0)

plotter.plot(r'C:\Users\artur\Desktop\30.09 milk\вода\дист_вода_эергия_12_%_Subt2__2__13-59-56-812.txt', -15000.0, 10)

plotter.draw_vertical_line(247.88, "C", False)

plotter.draw_vertical_line(253.57, "P")
plotter.draw_vertical_line(255.28)

plotter.draw_vertical_line(279.55, "Mg", False)
plotter.draw_vertical_line(285.21)

plotter.draw_vertical_line(393.33, "Ca")
plotter.draw_vertical_line(396.83)
plotter.draw_vertical_line(422.67, "Ca", False)
plotter.draw_vertical_line(558.87, "Ca")

plotter.draw_vertical_line(589.00, "Na")

plotter.draw_vertical_line(766.49, "K")
plotter.draw_vertical_line(769.90)

plotter.draw_vertical_line(656.26, "H")
plotter.draw_vertical_line(434.05, "H")

plotter.draw_vertical_line(777.35, "O")
plotter.draw_vertical_line(844.67, "O")

plotter.draw_vertical_line(746.89, "N")
plotter.draw_vertical_line(821.82, "N")
plotter.draw_vertical_line(868.28, "N")

plotter.draw_vertical_line(500.36, "N(II)")
plotter.draw_vertical_line(399.48, "N(II)", False)
plotter.draw_vertical_line(463.05, "N(II)", False)
plotter.draw_vertical_line(567.93, "N(II)", False)
plotter.draw_vertical_line(332.96, "N(II)", False)
plotter.draw_vertical_line(343.71)

# N еще смотри
# CN molecular bands

cursor = Cursor(ax, horizOn=False, vertOn=True, useblit=True, color='black', linewidth=1)
# plt.grid()
# plt.legend()

plt.show()
