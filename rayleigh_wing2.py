import math
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import cv2
from numba import jit
from scipy.signal import filtfilt
import tkinter as tk
from tkinter import filedialog
import os
import sys

# plt.tight_layout(pad=0)
analized_radius = 1000
freq = None
intensity = None
center = None
img_orig = None
fig, ax = plt.subplot_mosaic(
    """
    AB
    CC
    """
)


def lee_filter(img, size):
    img = np.array(img)
    img_mean = nd.uniform_filter(img, (size, size))
    img_sqr_mean = nd.uniform_filter(img ** 2, (size, size))
    img_variance = img_sqr_mean - img_mean ** 2

    overall_variance = nd.variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def img_intensity_normalize(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 1])
    accHist = np.cumsum(hist)
    threshold = 0.99 * np.max(accHist)
    v = next(i for i, x in enumerate(accHist) if x >= threshold) / 255.0
    newImg = img / v
    newImg = np.clip(newImg, 0.0, 1.0)
    # plt.imshow(newImg, cmap='gray', vmin=0, vmax=1)
    # plt.show()
    return newImg


def prepare_image(path, process=True):
    path = os.path.normpath(path)

    with open(path, "rb") as f:
        chunk = f.read()
    chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
    img_raw = cv2.imdecode(chunk_arr, cv2.IMREAD_GRAYSCALE)

    #img_raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    height, width = img_raw.shape
    img_raw = img_raw[0:height, 0:width]  # 420

    img = img_raw
    img = img.astype(np.float32) / 255.0  # Нормализуем в [0, 1]

    if process:
        img = img_intensity_normalize(img)
        img = nd.median_filter(img, size=5)
        img = lee_filter(img, 15)

        s = 2
        w = 5
        t = (((w - 1) / 2) - 0.5) / s
        img = nd.gaussian_filter(img, sigma=s, truncate=t)

    return img


def get_gradients(img):
    gradient_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=9)
    gradient_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=9)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2 + 1e-8)
    gradient_x /= gradient_magnitude
    gradient_y /= gradient_magnitude
    return gradient_x, gradient_y


def create_grids_xy(img):
    height, width = img.shape
    y_arr, x_arr = np.mgrid[0:height, 0:width]
    x_arr = x_arr.astype(np.float32)
    y_arr = y_arr.astype(np.float32)
    return x_arr, y_arr


@jit(nopython=True, parallel=True)
def get_error_map(gradients, grids, center):
    cx, cy = center
    gx, gy = gradients
    x_arr, y_arr = grids

    dx = x_arr - cx
    dy = y_arr - cy

    length_center_vec = np.sqrt(dx ** 2 + dy ** 2 + 1e-8)  # +1e-8 чтобы избежать деления на 0
    length_gradient = np.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    dot_product = dx * gx + dy * gy
    cos_angle = dot_product / (length_center_vec * length_gradient)

    error = 1 - np.abs(cos_angle)
    return error


@jit(nopython=True, parallel=True)
def error_value(gradients, grids, center):
    return np.sum(get_error_map(gradients, grids, center)) / 1000.0


def error_value_timer(gradients, grids, c):
    start_time = time.perf_counter()
    value = error_value(gradients, grids, c)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("Time: ", elapsed_time, "\nError value: ", value)
    return value


# @jit(nopython=True, parallel=True)
def get_sum_error_map_global(img, gradients, grids, step=50):
    width, height = img.shape
    xrange = list(range(0, width, step))
    yrange = list(range(0, height, step))
    errorMap = np.zeros((len(xrange), len(yrange)))
    xcounter = 0
    ycounter = 0
    print("total to do: ", len(xrange), len(yrange))

    globalCounter = 0
    time10 = 0
    time110 = 0
    start_time = time.perf_counter()
    for x in xrange:
        for y in yrange:
            errorMap[xcounter, ycounter] = error_value(gradients, grids, (y, x))
            ycounter += 1
            globalCounter += 1

            if globalCounter == 10:
                time10 = time.perf_counter()
            if globalCounter == 110:
                time110 = time.perf_counter()
                avg_time = (time110 - time10) / 100.0
                print("Avg time: ", avg_time, "\nEstimated for all calculations: ",
                      len(xrange) * len(yrange) * avg_time)
        xcounter += 1
        ycounter = 0

    end_time = time.perf_counter()
    print("Real total time: ", end_time - start_time)

    plt.subplot(2, 1, 1)
    im = plt.imshow(errorMap, cmap='gray')
    # plt.colorbar(im)
    plt.subplot(2, 1, 2)
    plt.imshow(img, cmap='gray')
    plt.show()
    return errorMap


def gradient_descent(img, gradients, grids, learning_rate=30.0, max_iter=100):
    width, height = img.shape

    current_coord = np.array([height / 2, width / 2], dtype=float)
    path = []

    for i in range(max_iter):
        x, y = current_coord
        path.append(current_coord)

        h = 1
        grad_x = (error_value(gradients, grids, [x + h, y]) - error_value(gradients, grids, [x - h, y])) / (2 * h)
        grad_y = (error_value(gradients, grids, [x, y + h]) - error_value(gradients, grids, [x, y - h])) / (2 * h)

        step = learning_rate * np.array([grad_x, grad_y])
        if np.linalg.norm(step) < 0.6:
            break
        new_coord = current_coord - step
        current_coord = np.round(new_coord).astype(int)
    path.append(current_coord)
    return current_coord, path


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
    for i in range(width):
        column = polar_img[:, i]
        value = column[column > 0].mean()
        rel_freq = (i / width) ** 2
        freq.append(rel_freq)
        intensity.append(value)

    n = 10  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    intensity = filtfilt(b, a, intensity)
    return freq, intensity


def center_find_pipeline(path):
    img = prepare_image(path)  # First index is height
    img_orig = prepare_image(path, False)
    gradients = get_gradients(img)
    grids = create_grids_xy(img)
    center, gradient_path = gradient_descent(img, gradients, grids, 50, 100)
    xc, yc = center
    center = (float(xc), float(yc))
    return img, img_orig, center, gradient_path


def redraw_spectre():
    global img_orig, center, ax, freq, intensity
    print(center)
    polar_img = polar_transform_cv2(img_orig, center, analized_radius, (analized_radius, int(analized_radius / 2)))
    ax['B'].clear()
    ax['B'].imshow(polar_img, cmap='gray')
    #ax['B'].minorticks_on()
    #ax['B'].grid(which='minor', alpha=0.2, color='green')
    ax['B'].grid(which='major', alpha=0.5, color='green')

    freq, intensity = create_spectre(polar_img)
    ax['C'].clear()
    ax['C'].plot(freq[20:], intensity[20:])
    plt.show()


def save_to_file():
    global path, freq, intensity, center
    dir = os.path.dirname(os.path.abspath(path))
    basename_without_ext = os.path.splitext(os.path.basename(path))[0]

    if not os.path.exists(dir + '\\specter'):
        os.makedirs(dir + '\\specter')

    save_path = dir + '\\specter\\' + basename_without_ext + '.txt'
    file = open(save_path, 'w')
    file.write('#center: ' + str(center) + '\n')
    file.write('#analized_radius: ' + str(analized_radius) + '\n')
    for i in range(len(freq)):
        file.write(str(freq[i]) + ' ' + str(intensity[i]) + '\n')
    print('Graph saved to file: ' + save_path)
    file.close()


def on_key_press(event):
    global center
    xc, yc = center

    if event.key == 'right':
        xc += 1
    elif event.key == 'left':
        xc -= 1
    elif event.key == 'up':
        yc -= 1
    elif event.key == 'down':
        yc += 1
    elif event.key == 'enter':
        save_to_file()

    center = (float(xc), float(yc))
    redraw_spectre()


root = tk.Tk()
root.withdraw()

path = filedialog.askopenfilename(initialdir=r'C:\Users\artur\Desktop')

# path = r'C:\Users\artur\Desktop\06.10 рэлей назад\дистиллят P=10% (2).png'
# path = r'C:\Users\artur\Desktop\06.10 рэлей назад\без обр 15 мкл P=50%.png'
# path = r'C:\Users\artur\Desktop\4 курс\Диплом 2025\circles\us_1\2.bmp'
img, img_orig, center, gradient_path = center_find_pipeline(path)

# center = (1096.0, 396.0) 06.10
# center = (948.0, 412.0) 02.10
# center = (1116.0, 956.0) 09.10

ax['A'].imshow(img, cmap='gray')
ax['A'].plot(*np.array(gradient_path).T, '.')

# 1095 405 center official / (1095, 398) unofficial better
# (float(x), float(y))

fig.canvas.mpl_connect('key_press_event', on_key_press)

redraw_spectre()
