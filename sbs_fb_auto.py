import math
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import cv2
from scipy.signal import argrelextrema
from numba import jit
from scipy.signal import filtfilt
import tkinter as tk
from scipy.optimize import curve_fit
from tkinter import filedialog
import os
import sys

base_width = 100

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from typing import Tuple, List, Optional


def fit_gaussian_to_peak(freq: np.ndarray, intensity: np.ndarray,
                         peak_center: float, window_size: float = 0.75) -> Tuple[float, float]:
    """
    Аппроксимирует отдельный пик гауссианом для точного определения центра.

    Parameters:
    -----------
    freq : np.ndarray
        Массив частот
    intensity : np.ndarray
        Массив интенсивностей
    peak_center : float
        Приблизительный центр пика
    window_size : float
        Размер окна вокруг пика для фита

    Returns:
    --------
    Tuple[float, float]
        Точный центр пика и ширина
    """
    # Выбираем окно вокруг пика
    mask = (freq >= peak_center - window_size) & (freq <= peak_center + window_size)
    if not np.any(mask):
        return peak_center, 0.01

    x_data = freq[mask]
    y_data = intensity[mask]

    # Начальное приближение
    amp_guess = np.max(y_data)
    cen_guess = x_data[np.argmax(y_data)]
    wid_guess = 0.1  # Увеличили начальную ширину

    try:
        popt, _ = curve_fit(gaussian, x_data, y_data,
                            p0=[amp_guess, cen_guess, wid_guess],
                            maxfev=2000)
        return popt[1], abs(popt[2])
    except:
        return cen_guess, wid_guess


def calibrate_frequency_scale(freq: np.ndarray, intensity: np.ndarray,
                              fsr: float,
                              method: str = 'gaussian',  # Оставляем 'gaussian' как основной метод
                              plot: bool = False,
                              min_peak_height: float = 0.06) -> Tuple[np.ndarray, float, List[float]]:
    """
    Калибрует частотную шкалу с использованием FSR.

    Parameters:
    -----------
    freq : np.ndarray
        Исходная частотная шкала (условные единицы)
    intensity : np.ndarray
        Интенсивность спектра
    fsr : float
        Область свободной дисперсии в реальных единицах (например, ГГц)
    method : str
        Метод определения пиков: 'gaussian' или 'max'
    plot : bool
        Построить график для визуализации

    Returns:
    --------
    Tuple[np.ndarray, float, List[float]]
        - Калиброванная частотная шкала
        - Коэффициент пересчета (реальные единицы / условные единицы)
        - Центры пиков в условных единицах
    """
    from scipy.signal import argrelextrema

    # Сначала вычитаем baseline с использованием существующей функции
    intensity_corrected = subtract_baseline(freq, intensity, order=5, plot=False)

    # Находим пики через argrelextrema (всегда используем этот метод для поиска)
    all_peak_indices = argrelextrema(intensity_corrected, np.greater, order=10)[0]

    # Фильтруем пики по минимальной высоте
    valid_peaks = []
    for idx in all_peak_indices:
        if intensity_corrected[idx] >= min_peak_height:
            valid_peaks.append(idx)

    peak_indices = np.array(valid_peaks)
    peak_centers_rough = freq[peak_indices].tolist()

    if method == 'gaussian':
        # Уточняем позиции пиков через фит гауссианом
        peak_centers = []
        peak_widths = []

        for peak in peak_centers_rough:
            # Динамически определяем размер окна на основе расстояния между пиками
            if len(peak_centers_rough) > 1:
                # Находим ближайшие пики для определения окна
                distances = [abs(p - peak) for p in peak_centers_rough if p != peak]
                min_distance = min(distances) if distances else 1.5
                window_size = min_distance * 0.4  # 40% от расстояния до ближайшего пика
            else:
                window_size = 0.5  # значение по умолчанию

            center, width = fit_gaussian_to_peak(freq, intensity_corrected, peak,
                                                 window_size=window_size)
            peak_centers.append(center)
            peak_widths.append(width)
    else:
        # Используем грубые позиции пиков
        peak_centers = peak_centers_rough

    # Сортируем пики
    peak_centers = sorted(peak_centers)

    if len(peak_centers) < 2:
        raise ValueError(f"Найдено недостаточно пиков: {len(peak_centers)}. Нужно минимум 2.")

    # Вычисляем расстояния между соседними пиками
    peak_distances = np.diff(peak_centers)
    mean_distance = np.mean(peak_distances)

    # Коэффициент пересчета: FSR в реальных единицах / среднее расстояние в условных единицах
    conversion_factor = fsr / mean_distance

    # Калиброванная шкала
    freq_calibrated = freq * conversion_factor

    if plot:
        plt.figure(figsize=(12, 8))

        # Исходный спектр с вычтенным baseline
        plt.subplot(2, 1, 1)
        plt.plot(freq, intensity_corrected, 'b-', label='Спектр с вычтенным baseline')
        plt.plot(peak_centers, [intensity_corrected[np.abs(freq - p).argmin()] for p in peak_centers],
                 'ro', label='Найденные пики')
        plt.xlabel('Частота (усл. ед.)')
        plt.ylabel('Интенсивность')
        plt.title(f'Обнаружение пиков (FSR = {fsr:.1f} ед.)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Калиброванный спектр
        plt.subplot(2, 1, 2)
        plt.plot(freq_calibrated, intensity_corrected, 'g-')
        plt.xlabel(f'Частота (ГГц)')
        plt.ylabel('Интенсивность')
        plt.title('Калиброванный спектр')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Информация о калибровке
        print(f"Найдено пиков: {len(peak_centers)}")
        print(f"Позиции пиков (усл. ед.): {[f'{p:.4f}' for p in peak_centers]}")
        print(f"Расстояния между пиками (усл. ед.): {[f'{d:.4f}' for d in peak_distances]}")
        print(f"Среднее расстояние между пиками: {mean_distance:.4f} усл. ед.")
        print(f"FSR: {fsr} ГГц")
        print(f"Коэффициент пересчета: {conversion_factor:.4f} ГГц/усл.ед.")

    return freq_calibrated, conversion_factor, peak_centers


def get_calibrated_spectrum(filename: str, fsr: float,
                            crop_range: Tuple[float, float] = (0.06, 0.275),
                            plot: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Полная обработка: получение спектра из изображения и калибровка частотной шкалы.

    Parameters:
    -----------
    filename : str
        Путь к файлу изображения
    fsr : float
        Область свободной дисперсии в ГГц
    crop_range : Tuple[float, float]
        Диапазон частот для обрезки (мин, макс) в условных единицах
    plot : bool
        Построить графики

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, float]
        Калиброванные частоты, интенсивность, коэффициент пересчета
    """
    # Получаем спектр из изображения
    img = prepare_image(filename)
    analized_radius = 145
    polar_img = polar_transform_cv2(img, (150.25, 147.0), analized_radius,
                                    (analized_radius * 2, int(analized_radius)))
    freq, intensity = create_spectre(polar_img)

    # Обрезаем спектр
    freq, intensity = crop_spectre(freq, intensity, crop_range[0], crop_range[1])

    # Калибруем частотную шкалу
    freq_calibrated, conversion_factor, peak_centers = calibrate_frequency_scale(
        freq, intensity, fsr, method='gaussian', plot=plot
    )

    return freq_calibrated, intensity, conversion_factor


# =====================================================================================


def lee_filter(img, size):
    img = np.array(img)
    img_mean = nd.uniform_filter(img, (size, size))
    img_sqr_mean = nd.uniform_filter(img ** 2, (size, size))
    img_variance = img_sqr_mean - img_mean ** 2

    overall_variance = nd.variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def prepare_image(path, process=True):
    img_raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    height, width = img_raw.shape
    img_raw = img_raw[0:height, 0:width]  # 420

    img = img_raw
    img = img.astype(np.float32) / 255.0  # Нормализуем в [0, 1]

    if process:
        img = lee_filter(img, 5)

    return img


def polar_transform_cv2(image, center, radius, output_shape=None):
    if output_shape is None:
        output_height = radius
        output_width = int(2 * math.pi * radius)
    else:
        output_width, output_height = output_shape

    # Используем встроенную функцию OpenCV для линейного полярного преобразования
    polar_image = cv2.warpPolar(
        image,
        (output_width, output_height),
        center,
        radius,
        cv2.WARP_POLAR_LINEAR
    )

    return polar_image


def create_spectre(polar_img):
    height, width = polar_img.shape

    intensity = []
    freq = []

    # y_start = int(height/2)
    # y_end = y_start + 10
    y_start = 80
    y_end = y_start + 4
    for i in range(width):
        column = polar_img[y_start:y_end, i]
        value = column[column > 0].mean()
        rel_freq = (i / base_width) ** 2
        freq.append(rel_freq)
        intensity.append(value)

    n = 8  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    intensity = filtfilt(b, a, intensity)
    return np.array(freq), np.array(intensity)


def crop_spectre(freq, intensity, min_v, max_v):
    mask = (freq >= min_v) & (freq <= max_v)
    return freq[mask], intensity[mask]


def find_local_minima(freq: np.ndarray, intensity: np.ndarray,
                      order: int = 5) -> [np.ndarray, np.ndarray]:
    """
    Находит локальные минимумы в спектре с учетом неравномерной сетки частот.

    Parameters:
    -----------
    freq : np.ndarray
        Массив частот (может быть неравномерным)
    intensity : np.ndarray
        Массив интенсивностей
    order : int
        Параметр сглаживания для поиска минимумов (чем больше, тем менее чувствителен)

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Частоты и интенсивности в точках локальных минимумов
    """
    # Находим индексы локальных минимумов
    # Используем argrelextrema для поиска минимумов
    min_idx = argrelextrema(intensity, np.less, order=order)[0]

    # Добавляем граничные точки, если они не являются минимумами
    if len(min_idx) == 0:
        # Если минимумов нет, используем граничные точки
        min_idx = np.array([0, len(freq) - 1])
    else:
        # Проверяем первую точку
        if intensity[0] <= intensity[1]:
            min_idx = np.insert(min_idx, 0, 0)
        # Проверяем последнюю точку
        if intensity[-1] <= intensity[-2]:
            min_idx = np.append(min_idx, len(freq) - 1)

    return freq[min_idx], intensity[min_idx]


def subtract_baseline(freq: np.ndarray, intensity: np.ndarray,
                      order: int = 5, plot: bool = False) -> np.ndarray:
    """
    Вычитает базовую линию, соединяющую первый и последний локальный минимум.

    Parameters:
    -----------
    freq : np.ndarray
        Массив частот
    intensity : np.ndarray
        Массив интенсивностей
    order : int
        Параметр для поиска минимумов
    plot : bool
        Если True, показывает график для визуализации

    Returns:
    --------
    np.ndarray
        Интенсивность с вычтенным baseline
    """
    # Находим локальные минимумы
    min_freq, min_intensity = find_local_minima(freq, intensity, order)

    if len(min_freq) < 2:
        # Если недостаточно минимумов, используем первую и последнюю точки
        baseline = np.poly1d(np.polyfit([freq[0], freq[-1]],
                                        [intensity[0], intensity[-1]], 1))(freq)
    else:
        # Используем первый и последний локальный минимум
        x_baseline = [min_freq[0], min_freq[-1]]
        y_baseline = [min_intensity[0], min_intensity[-1]]

        # Строим линейную интерполяцию
        coeffs = np.polyfit(x_baseline, y_baseline, 1)
        baseline = np.poly1d(coeffs)(freq)

    # Вычитаем baseline
    corrected_intensity = intensity - baseline

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(freq, intensity, 'b-', label='Исходный спектр')
        plt.plot(freq, baseline, 'r--', label='Базовая линия')
        plt.plot(min_freq, min_intensity, 'ro', label='Локальные минимумы')
        plt.plot(freq, corrected_intensity, 'g-', label='Спектр после вычитания')
        plt.xlabel('Частота')
        plt.ylabel('Интенсивность')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return corrected_intensity


def nullify_wings(freq: np.ndarray, intensity: np.ndarray,
                  order: int = 5) -> [np.ndarray, np.ndarray]:
    # Находим все локальные минимумы
    min_freq, min_intensity = find_local_minima(freq, intensity, order)

    if len(min_freq) < 2:
        print("Предупреждение: Найдено менее 2 локальных минимумов. Обрезка не выполняется.")
        return freq.copy(), intensity.copy()

    # Определяем границы обрезки (первый и последний минимум)
    left_bound = min_freq[0]
    right_bound = min_freq[-1]

    # Создаем маску для обрезки
    mask = (freq <= left_bound) | (freq >= right_bound)
    intensity[mask] = 0.0

    return intensity


def fit_two_gaussians(freq: np.ndarray, intensity: np.ndarray,
                      initial_guess: [float, float, float, float, float, float],
                      plot: bool = False) -> [np.ndarray, np.ndarray]:
    """
    Аппроксимирует спектр суммой двух гауссовых профилей.

    Parameters:
    -----------
    freq : np.ndarray
        Массив частот
    intensity : np.ndarray
        Массив интенсивностей (после вычитания baseline)
    initial_guess : tuple
        Начальное приближение в формате:
        (amp1, cen1, wid1, amp2, cen2, wid2)
    plot : bool
        Если True, показывает график результата

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Оптимальные параметры и ковариационная матрица
    """
    try:
        # Выполняем подгонку
        popt, pcov = curve_fit(two_gaussians, freq, intensity,
                               p0=initial_guess, maxfev=5000)

        if plot:
            # Генерируем точки для гладкой кривой
            freq_smooth = np.linspace(freq.min(), freq.max(), 500)
            fit_total = two_gaussians(freq_smooth, *popt)
            fit1 = gaussian(freq_smooth, popt[0], popt[1], popt[2])
            fit2 = gaussian(freq_smooth, popt[3], popt[4], popt[5])

            plt.figure(figsize=(10, 6))
            plt.plot(freq, intensity, 'bo', markersize=3, label='Данные')
            plt.plot(freq_smooth, fit_total, 'r-', label='Суммарная подгонка')
            plt.plot(freq_smooth, fit1, 'g--', label='Гауссиан 1')
            plt.plot(freq_smooth, fit2, 'm--', label='Гауссиан 2')
            plt.xlabel('Частота')
            plt.ylabel('Интенсивность')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title('Аппроксимация суммой двух гауссианов')
            plt.show()

        return popt

    except Exception as e:
        print(f"Ошибка при подгонке: {e}")
        x = -10000.0
        return [x, x, x, x, x, x]


def gaussian(x: np.ndarray, amp: float, cen: float, wid: float) -> np.ndarray:
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))


def two_gaussians(x: np.ndarray, amp1: float, cen1: float, wid1: float,
                  amp2: float, cen2: float, wid2: float) -> np.ndarray:
    return (gaussian(x, amp1, cen1, wid1) +
            gaussian(x, amp2, cen2, wid2))


def get_freq_koeff(filename: str, method='gaussian', plot=False):
    img = prepare_image(filename)
    analized_radius = 145
    polar_img = polar_transform_cv2(img, (150.25, 147.0), analized_radius, (analized_radius * 2, int(analized_radius)))
    freq, intensity = create_spectre(polar_img)

    fsr = 300.0 / 12.0 / 2.0  # GHz
    print(f"File: {filename}")

    freq_calibrated, koeff, _ = calibrate_frequency_scale(freq, intensity, fsr=fsr, method=method, plot=plot)
    if not plot:
        # print(f'FSR: {fsr} GHz')
        print(f'Коэффициент пересчета: {koeff:.4f} GHz/усл.ед.')
    return koeff


def avg_freq_koeff(base_folder: str):
    koeffs = []
    path_arr = []

    for subdir, dirs, files in os.walk(base_folder):
        for file in files:

            path = base_folder + '\\' + file
            if "лазер" in file:
                path_arr.append(path)

    for path in path_arr:
        koeffs.append(get_freq_koeff(path, 'gaussian', False))

    print(f"Avg koeff: {np.average(koeffs):.4f}, std.dev: {np.std(koeffs):.4f}")
    return np.average(koeffs)


def view_params(filename: str, freq_koeff: float):
    img = prepare_image(filename)
    analized_radius = 145
    polar_img = polar_transform_cv2(img, (150.25, 147.0), analized_radius, (analized_radius * 2, int(analized_radius)))
    freq, intensity = create_spectre(polar_img)
    freq *= freq_koeff
    #plt.plot(freq, intensity)
    #plt.show()
    freq, intensity = crop_spectre(freq, intensity, 0.45 * freq_koeff, 2.25 * freq_koeff)
    return freq, intensity


def return_spectre_params(popt, filename, plot_fit):
    laser_freq, laser_width = popt[1], popt[2]
    sbs_freq, sbs_width = popt[4], popt[5]
    freq_shift = 12.5 - (sbs_freq - laser_freq)

    laser_width = abs(laser_width)
    sbs_width = abs(sbs_width)

    if plot_fit:
        print(f"Fit result: {filename}")
        print(f"Frequency shift: {freq_shift:.4f}")
        print(f"Laser width: {laser_width:.4f}")
        print(f"SBS width: {sbs_width:.4f}")
        print(f"Intensities minimum: {popt[0]:.4f}, {popt[3]:.4f}")
    return laser_width, sbs_width, freq_shift, popt[0], popt[3]


def get_spectre_params(filename: str, freq_koeff: float, plot_fit=False, order: int = 0):
    img = prepare_image(filename)
    analized_radius = 145
    polar_img = polar_transform_cv2(img, (150.25, 147.0), analized_radius, (analized_radius * 2, int(analized_radius)))
    freq, intensity = create_spectre(polar_img)
    freq *= freq_koeff
    # print(order)
    if order == 0:
        # print("order == 0")
        freq, intensity = crop_spectre(freq, intensity, 0.45 * freq_koeff, 2.25 * freq_koeff)
    elif order == 1:
        # print("order == 1")
        freq, intensity = crop_spectre(freq, intensity, 2.0 * freq_koeff, 3.8 * freq_koeff)
        freq -= 1.55 * freq_koeff
    else:
        # print("else order")
        freq, intensity = crop_spectre(freq, intensity, 3.3 * freq_koeff, 5.35 * freq_koeff)
        freq -= 2.85 * freq_koeff

    intensity = subtract_baseline(freq, intensity,
                                  order=5,
                                  plot=False)

    intensity = nullify_wings(freq, intensity, order=5)

    # amp, cen, wid
    popt = fit_two_gaussians(freq, intensity,
                             [0.1, 1.06 * freq_koeff, 0.4 * freq_koeff, 0.1, 1.7 * freq_koeff, 0.4 * freq_koeff],
                             plot_fit)

    if popt[0] > 0.0 and popt[3] > 0.0:
        return return_spectre_params(popt, filename, plot_fit)

    # second try
    popt = fit_two_gaussians(freq, intensity,
                             [0.2, 0.93 * freq_koeff, 0.25 * freq_koeff, 0.06, 1.56 * freq_koeff, 0.25 * freq_koeff],
                             plot_fit)

    if popt[0] > 0.0 and popt[3] > 0.0:
        return return_spectre_params(popt, filename, plot_fit)

    # third try
    popt = fit_two_gaussians(freq, intensity,
                             [0.09, 1.08 * freq_koeff, 0.25 * freq_koeff, 0.105, 1.7 * freq_koeff, 0.25 * freq_koeff],
                             plot_fit)

    if popt[0] > 0.0 and popt[3] > 0.0:
        return return_spectre_params(popt, filename, plot_fit)


    print("Unable to perform normal fit")
    return -100.0, -100.0, -100.0, -100.0, -100.0



def get_all_specters_with_depth(base_folder: str, depth_str: str):
    path_arr = []

    for subdir, dirs, files in os.walk(base_folder):
        for file in files:

            if not depth_str in file[:len(depth_str)]:
                continue

            if not '.tif' in file:
                continue

            if "rev_wires" in file:
                continue

            path = base_folder + '\\' + file
            path_arr.append(path)
    return path_arr


def print_array(x: np.array):
    print(f"[", end='')
    for i in range(len(x)):
        if i != len(x) - 1:
            print(f"{x[i]:.2f}, ", end='')
        else:
            print(f"{x[i]:.2f}", end='')
    print(f"]")


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


def get_avg_params_for_width(base_folder: str, depth_str: str, freq_koeff: float, orders=[0], use_median: bool = True):
    laser_arr = []
    sbs_arr = []
    shift_arr = []

    depth_float = float(depth_str[:-2])

    path_arr = get_all_specters_with_depth(base_folder, depth_str)
    for order in orders:
        for path in path_arr:
            laser_width, sbs_width, freq_shift, i0, i1 = get_spectre_params(path, freq_koeff, False, order)

            if laser_width < 0.0:
                print(f"Unable to process {path}")
                continue

            if freq_shift > 8.5 or freq_shift < 6.5:
                continue

            if min(i0, i1) < 0.02:
                continue

            i_ratio = i0 / i1
            if i_ratio < 1.0:
                i_ratio = 1.0 / i_ratio

            if i_ratio > 7.0:
                continue

            laser_arr.append(laser_width)
            sbs_arr.append(sbs_width)
            shift_arr.append(freq_shift)

    print(f"==================== Depth {depth_str} ====================")
    print(f"Laser width mean: {np.mean(laser_arr):.3f}, std.dev: {np.std(laser_arr):.3f}")
    print_array(laser_arr)
    print(f"SBS width mean: {np.mean(sbs_arr):.3f}, std.dev: {np.std(sbs_arr):.3f}")
    print_array(sbs_arr)
    print(f"Freq. shift mean: {np.mean(shift_arr):.3f}, std.dev: {np.std(shift_arr):.3f}")
    print_array(shift_arr)
    if use_median:
        return depth_float, np.median(laser_arr), IQR(laser_arr), np.median(sbs_arr), IQR(sbs_arr), np.median(
            shift_arr), IQR(shift_arr)
    else:
        return depth_float, np.mean(laser_arr), np.std(laser_arr), np.mean(sbs_arr), np.std(sbs_arr), np.mean(
            shift_arr), np.std(shift_arr)


def all_depth_analysis(base_folder: str, freq_koeff: float,  order: int = 0):
    depths = []
    laser_width_arr = []
    laser_error_arr = []
    sbs_width_arr = []
    sbs_error_arr = []
    shift_arr = []
    shift_error_arr = []

    def add(depth_str):
        depth, laser_width, laser_error, sbs_width, sbs_error, shift, shift_error = get_avg_params_for_width(
            base_folder, depth_str, freq_koeff, order)

        depths.append(depth)
        laser_width_arr.append(laser_width)
        laser_error_arr.append(laser_error)
        sbs_width_arr.append(sbs_width)
        sbs_error_arr.append(sbs_error)
        shift_arr.append(shift)
        shift_error_arr.append(shift_error)

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
    plt.errorbar(depths, laser_width_arr, yerr=laser_error_arr, label='Лазер ширина', ecolor='gray', fmt='o-')
    plt.errorbar(depths, sbs_width_arr, yerr=sbs_error_arr, label='ВРМБ ширина', ecolor='gray', fmt='o-')
    plt.xlabel("Глубина, мм")
    plt.ylabel("Частота, ГГц")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12.8, 9.6))
    plt.errorbar(depths, shift_arr, yerr=shift_error_arr, label='Сдвиг частоты', ecolor='gray', fmt='o-')
    plt.xlabel("Глубина, мм")
    plt.ylabel("Частота, ГГц")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# Avg koeff: 8.4091, std.dev: 0.0109 - 'gaussian'
# Avg koeff: 8.4082, std.dev: 0.0249 - 'max'
plt.rcParams.update({'font.size': 16})
koeff = avg_freq_koeff('C:\\Users\\artur\\Desktop\\13.02.26 sbs new\\fb_crop\\')
# all_depth_analysis('C:\\Users\\artur\\Desktop\\13.02.26 sbs new\\fb_crop\\', koeff, [0, 1])

f1, i1 = view_params('C:\\Users\\artur\\Desktop\\13.02.26 sbs new\\fb_crop\\2.5mm3.tif', koeff)
i1 /= i1.max()
f2, i2 = view_params('C:\\Users\\artur\\Desktop\\13.02.26 sbs new\\fb_crop\\5mm7.tif', koeff)
i2 /= i2.max()
f2 += 0.03 * koeff
plt.plot(f2, i2)
plt.plot(f1, i1)
plt.show()

# view_params('C:\\Users\\artur\\Desktop\\13.02.26 sbs new\\fb_crop\\1mm3.tif', 1.0)
# get_spectre_params('C:\\Users\\artur\\Desktop\\13.02.26 sbs new\\fb_crop\\1mm3.tif', 1.0, plot_fit=True, order=0)

# 1mm4.tif - bad

# get_avg_params_for_width('C:\\Users\\artur\\Desktop\\13.02.26 sbs new\\fb_crop\\', '0.6mm', koeff)
