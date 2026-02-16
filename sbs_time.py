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


def read_file(filename: str) -> [np.array, np.array, np.array]:
    file = open(filename)
    t = []
    ch1 = []
    ch2 = []

    # Read file
    for line in file:
        if line[0].isnumeric() or line[0] == '-':
            values = line.split(',')

            time = float(values[0]) * 1e9
            t.append(time)
            ch1.append(float(values[1]))
            ch2.append(float(values[2]))

    return np.array(t), np.array(ch1), np.array(ch2)


def process_y(y: np.array):
    y = filter_noise(y)
    y = remove_background(y)
    # y = normalize(y)
    return y


def process_oscilloscope_data(t, ch1, ch2, trigger_value=0.0, pre_trigger=-30.0, post_trigger=70.0):
    """
    Обработка данных осциллографа: имитация триггера и обрезка по времени.

    Параметры:
    -----------
    t : np.array
        Временные метки
    ch1 : np.array
        Данные первого канала
    ch2 : np.array
        Данные второго канала
    trigger_value : float
        Значение напряжения, при котором срабатывает триггер
    pre_trigger : float
        Время до триггера (отрицательное)
    post_trigger : float
        Время после триггера (положительное)

    Возвращает:
    -----------
    tuple: (t_trimmed, ch1_trimmed, ch2_trimmed, trigger_idx)
        Обрезанные массивы времени и каналов, а также индекс точки триггера
    """

    # Проверка входных данных
    if len(t) == 0 or len(ch1) == 0 or len(ch2) == 0:
        raise ValueError("Входные массивы не могут быть пустыми")

    if len(t) != len(ch1) or len(t) != len(ch2):
        raise ValueError("Все массивы должны иметь одинаковую длину")

    # Поиск точки триггера на возрастающем фронте
    trigger_idx = find_trigger_rising_edge(t, ch2, trigger_value)

    if trigger_idx is None:
        # Если триггер не найден, используем первый индекс
        trigger_idx = 0
        print("Предупреждение: Точка триггера не найдена, используется начало данных")

    # Сдвиг времени так, чтобы t=0 соответствовал моменту триггера
    t_shifted = t - t[trigger_idx]

    # Обрезка данных по временному интервалу
    mask = (t_shifted >= pre_trigger) & (t_shifted <= post_trigger)

    # Проверка, что маска не пустая
    if not np.any(mask):
        raise ValueError(f"Нет данных в интервале [{pre_trigger}, {post_trigger}]")

    t_trimmed = t_shifted[mask]
    ch1_trimmed = ch1[mask]
    ch2_trimmed = ch2[mask]

    return t_trimmed, ch1_trimmed, ch2_trimmed


def find_trigger_rising_edge(t, signal, threshold):
    """
    Поиск точки триггера на возрастающем фронте сигнала.

    Параметры:
    -----------
    t : np.array
        Временные метки
    signal : np.array
        Сигнал канала
    threshold : float
        Пороговое значение триггера

    Возвращает:
    -----------
    int or None: Индекс точки триггера или None, если не найдена
    """

    # Находим точки, где сигнал пересекает порог снизу вверх
    crosses = np.where((signal[:-1] < threshold) & (signal[1:] >= threshold))[0]

    if len(crosses) == 0:
        return None

    # Берем первое пересечение
    cross_idx = crosses[0]

    # Интерполируем для более точного определения момента пересечения
    t1, t2 = t[cross_idx], t[cross_idx + 1]
    v1, v2 = signal[cross_idx], signal[cross_idx + 1]

    # Линейная интерполяция для нахождения точного времени пересечения
    if v2 != v1:
        t_trigger = t1 + (threshold - v1) * (t2 - t1) / (v2 - v1)
    else:
        t_trigger = t1

    # Находим ближайший индекс к вычисленному времени триггера
    trigger_idx = np.argmin(np.abs(t - t_trigger))

    return trigger_idx


def process_all(t: np.array, ch1: np.array, ch2: np.array):
    ch1 = process_y(ch1)
    ch2 = process_y(ch2)
    t, ch1, ch2 = process_oscilloscope_data(t, ch1, ch2, 0.15, -20.0, 50.0)
    return t, ch1, ch2


def normalize(y: np.array):
    return y / np.max(y)


def remove_background(y: np.array):
    background = np.average(y[:100])
    return y - background


def filter_noise(y, n=6):
    b = [1.0 / n] * n
    a = 1
    return filtfilt(b, a, y)


def is_good_signal(t: np.array, ch1: np.array, ch2: np.array):
    # Всегда было 0.4 когда строил графики
    if np.max(ch2) < 0.5:
        return False

    return True


def naive_fwhm(t: np.array, ch: np.array) -> [float, float]:
    max_val = np.max(ch)
    half_max = max_val / 2

    # Находим все точки, где сигнал превышает половину максимума
    above_half = ch >= half_max

    if not np.any(above_half):
        # Если нет точек выше половины максимума
        print("Предупреждение: Сигнал не достигает половины максимума")
        return 0.0, t[0] if len(t) > 0 else 0.0

    # Находим индексы начала и конца области выше половины максимума
    indices = np.where(above_half)[0]
    start_idx = indices[0]
    end_idx = indices[-1]

    # Уточняем время начала с помощью интерполяции
    if start_idx > 0:
        # Линейная интерполяция между точкой до начала и точкой начала
        t1, t2 = t[start_idx - 1], t[start_idx]
        v1, v2 = ch[start_idx - 1], ch[start_idx]

        if v2 != v1:
            start_time = t1 + (half_max - v1) * (t2 - t1) / (v2 - v1)
        else:
            start_time = t1
    else:
        # Если начало области совпадает с началом массива
        start_time = t[start_idx]

    # Уточняем время конца с помощью интерполяции
    if end_idx < len(t) - 1:
        # Линейная интерполяция между точкой конца и следующей точкой
        t1, t2 = t[end_idx], t[end_idx + 1]
        v1, v2 = ch[end_idx], ch[end_idx + 1]

        if v1 != v2:
            end_time = t1 + (half_max - v1) * (t2 - t1) / (v2 - v1)
        else:
            end_time = t1
    else:
        # Если конец области совпадает с концом массива
        end_time = t[end_idx]

    # Вычисляем FWHM
    fwhm = end_time - start_time

    return fwhm, start_time


# Exponentially modified Gaussian
def emg(t, I0, t0, sigma, tau):
    # Защита от деления на ноль и отрицательных tau
    tau = np.abs(tau) + 1e-10

    # Вычисляем EMG
    lambda_param = 1.0 / tau
    arg = (lambda_param * sigma ** 2 + (t0 - t)) / (np.sqrt(2) * sigma)

    # Используем erfc для численной стабильности
    y = I0 * 0.5 * lambda_param * np.exp(lambda_param * (t0 - t) + 0.5 * (lambda_param * sigma) ** 2) * erfc(
        arg / np.sqrt(2))

    # Альтернативная формула (иногда более стабильная):
    # y = I0 * 0.5 * lambda_param * np.exp(0.5 * (lambda_param * sigma)**2 - lambda_param * (t - t0)) * erfc((lambda_param * sigma**2 - (t - t0))/(np.sqrt(2) * sigma))

    return y


def emg_fit(t: np.array, ch: np.array) -> [np.array, np.array]:
    try:
        popt, _ = curve_fit(emg, t, ch, p0=[30.0, 6.0, 4.0, 8.0])
    except:
        return None, None

    ch_fit = emg(t, *popt)
    return t, ch_fit


def naive_fwhm_two_channels(t: np.array, ch1: np.array, ch2: np.array):
    fwhm1, start_time1 = naive_fwhm(t, ch1)
    fwhm2, start_time2 = naive_fwhm(t, ch2)
    delay = start_time2 - start_time1
    return fwhm1, fwhm2, delay


def read_file_and_process(filename: str) -> [np.array, np.array, np.array]:
    data = read_file(filename)
    data = process_all(*data)
    return data


def print_array(x: np.array):
    print(f"[", end='')
    for i in range(len(x)):
        if i != len(x) - 1:
            print(f"{x[i]:.1f}, ", end='')
        else:
            print(f"{x[i]:.1f}", end='')
    print(f"]")


def get_all_csv_with_depth(base_folder: str, depth_str: str):
    path_arr = []

    for subdir, dirs, files in os.walk(base_folder):
        for file in files:

            if not depth_str in file[:len(depth_str)]:
                continue

            if not 'csv' in file:
                continue

            if "rev_wires" in file:
                continue

            path = base_folder + '\\' + file
            path_arr.append(path)
    return path_arr


def trim_percentile(arr, percent):
    """
    Удаляет указанный процент самых маленьких и самых больших значений

    Parameters:
    arr: входной массив
    percent: процент значений для удаления с каждого края (от 0 до 50)
    """
    if percent < 0 or percent >= 50:
        raise ValueError("Percent должен быть в диапазоне [0, 50)")

    # Сортируем массив
    sorted_arr = np.sort(arr)

    # Вычисляем количество элементов для удаления с каждого края
    n = len(arr)
    k = int(n * percent / 100)

    # Возвращаем массив без крайних значений
    return sorted_arr[k:n - k]


def MAD(data):
    """
    Calculate the Median Absolute Deviation (MAD) of a NumPy array.
    """
    data_median = np.median(data)
    abs_deviations = np.absolute(data - data_median)
    mad = np.median(abs_deviations)
    return mad


def IQR(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr_value = q3 - q1
    return iqr_value


def analyze_all_with_depth(base_folder: str, depth_str: str, print_result=True, use_emg=False, use_median=False):
    laser_fwhm = []
    sbs_fwhm = []
    delays = []

    depth_float = float(depth_str[:-2])

    for path in get_all_csv_with_depth(base_folder, depth_str):
        t, ch1, ch2 = read_file_and_process(path)
        if not is_good_signal(t, ch1, ch2):
            continue

        if use_emg:
            _, ch1 = emg_fit(t, ch1)
            _, ch2 = emg_fit(t, ch2)
        if ch1 is None or ch2 is None:
            continue

        fwhm1, fwhm2, delay = naive_fwhm_two_channels(t, ch1, ch2)
        print(f"{path}: fwhm1:{fwhm1:.1f}, fwhm2:{fwhm2:.1f}, delay:{delay:.1f}")
        laser_fwhm.append(fwhm1)
        sbs_fwhm.append(fwhm2)
        delays.append(delay)

    use_trim_percentile = False
    if use_trim_percentile:
        percentile = 15
        laser_fwhm = trim_percentile(laser_fwhm, percentile)
        sbs_fwhm = trim_percentile(sbs_fwhm, percentile)
        delays = trim_percentile(delays, percentile)

    if print_result:
        print(f"\n========= Depth {depth_str} =========")
        print(f"!!! Use EMG: {use_emg} !!!")
        print(f"Laser FWHM. Average: {np.average(laser_fwhm):.1f}, Std.Dev: {np.std(laser_fwhm):.1f}")
        print_array(laser_fwhm)
        print(f"\nSBS FWHM. Average: {np.average(sbs_fwhm):.1f}, Std.Dev: {np.std(sbs_fwhm):.1f}")
        print_array(sbs_fwhm)
        print(f"\nSignal delay. Average: {np.average(delays):.1f}, Std.Dev: {np.std(delays):.1f}")
        print_array(delays)

    if use_median:
        return depth_float, np.median(laser_fwhm), IQR(laser_fwhm), np.median(sbs_fwhm), IQR(sbs_fwhm), np.median(
            delays), IQR(delays)
    else:
        return depth_float, np.average(laser_fwhm), np.std(laser_fwhm), np.average(sbs_fwhm), np.std(sbs_fwhm), np.average(
            delays), np.std(delays)


def plot_all_fwhm_and_delays(base_folder, use_emg=False, use_median=False):
    depths = []
    laser_fwhms = []
    laser_errors = []
    sbs_fwhms = []
    sbs_errors = []
    delays = []
    delay_errors = []

    def add(depth_str):
        depth, laser_fwhm, laser_error, sbs_fwhm, sbs_error, delay, delay_error = analyze_all_with_depth(base_folder,
                                                                                                         depth_str,
                                                                                                         True, use_emg, use_median)
        depths.append(depth)
        laser_fwhms.append(laser_fwhm)
        laser_errors.append(laser_error)
        sbs_fwhms.append(sbs_fwhm)
        sbs_errors.append(sbs_error)
        delays.append(delay)
        delay_errors.append(delay_error)

    add('0mm')
    add('0.3mm')
    add('0.6mm')
    add('1mm')
    add('1.5mm')
    add('2mm')
    add('2.5mm')
    add('3.5mm')
    add('5mm')
    add('10mm')

    plt.figure(figsize=(12.8, 9.6))
    plt.errorbar(depths, laser_fwhms, yerr=laser_errors, label='Лазер длительность', ecolor='gray', fmt='o-')
    plt.errorbar(depths, sbs_fwhms, yerr=sbs_errors, label='ВРМБ длительность', ecolor='gray', fmt='o-')
    plt.errorbar(depths, delays, yerr=delay_errors, label='Задержка', ecolor='gray', fmt='o-')
    plt.xlabel("Глубина, мм")
    plt.ylabel("Время, нс")
    plt.legend()
    plt.grid()
    plt.title(f'use_emg={use_emg}, use_median={use_median}')
    plt.tight_layout()
    plt.show()


def plot_all_with_depth(base_folder: str, depth_str: str, color1='', color2=''):
    for path in get_all_csv_with_depth(base_folder, depth_str):
        t, ch1, ch2 = read_file_and_process(path)
        if not is_good_signal(t, ch1, ch2):
            continue

        if color1 != '':
            plt.plot(t, ch1, color=color1, zorder=-1)
        if color2 != '':
            plt.plot(t, ch2, color=color2, zorder=0)


def example_plot(filename: str):
    t, ch1, ch2 = read_file_and_process(filename)
    file_name_with_ext = Path(filename).name

    # plt.plot(*emg_fwhm(t, ch1))

    plt.figure(figsize=(12.8, 9.6))
    plt.plot(t, ch1, label='Лазер', color='skyblue')
    plt.plot(t, ch2, label='ВРМБ', color='peachpuff')
    # plt.plot(t, emg(t, 30.0, 6.0, 4.0, 8.0))
    plt.plot(*emg_fit(t, ch1), label='Лазер fit', color='royalblue')
    plt.plot(*emg_fit(t, ch2), label='ВРМБ fit', color='coral')
    # plt.plot(t, emg(t, 25.0, 5.5, 3.5, 7.0))
    plt.ylabel("Сигнал, усл.ед.")
    plt.xlabel("Время, нс")
    plt.grid()
    plt.title(file_name_with_ext)
    plt.legend()
    plt.tight_layout()
    plt.show()


# color1='palegreen', color2='grey'
# color1='cyan', color2='pink'
print(f"Time in nanoseconds!")

plt.rcParams.update({'font.size': 16})

base_path = 'C:\\Users\\artur\\Desktop\\13.02.26 sbs new\\'
#example_plot(base_path + '10mm_011_ALL.csv')

plot_all_fwhm_and_delays(base_path, use_emg=True, use_median=True)

'''
analyze_all_with_depth(base_path, '0mm')
analyze_all_with_depth(base_path, '0.3mm')
analyze_all_with_depth(base_path, '0.6mm')
analyze_all_with_depth(base_path, '1mm')
analyze_all_with_depth(base_path, '1.5mm')
analyze_all_with_depth(base_path, '2mm')
analyze_all_with_depth(base_path, '2.5mm')
analyze_all_with_depth(base_path, '3.5mm')
analyze_all_with_depth(base_path, '5mm')
analyze_all_with_depth(base_path, '10mm')
'''
