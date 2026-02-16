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


def read_dls_txt(filename: str) -> [np.array, np.array]:
    file = open(filename)
    nm = []
    intensity = []

    for line in file:
        if len(line) < 2:
            continue
        if not line[1].isnumeric():
            continue

        values = line.split('\t\t')
        if len(values) != 2:
            continue
        nm.append(float(values[0]))
        intensity.append(float(values[1]))

    return np.array(nm), np.array(intensity)


def filter_noise(y, n=6):
    b = [1.0 / n] * n
    a = 1
    return filtfilt(b, a, y)


def plot_dls_txt(filename: str, color=''):
    nm, intensity = read_dls_txt(filename)
    file_name_with_ext = Path(filename).name

    # plt.figure(figsize=(12.8, 9.6))
    if color == '':
        plt.plot(nm, intensity, label=file_name_with_ext)
    else:
        plt.plot(nm, intensity, label=file_name_with_ext, color=color)
    plt.xscale('log')
    plt.ylabel("Интенсивность рассеяния")
    plt.xlabel("Размер частиц, нм")
    plt.grid()
    # plt.legend()
    plt.tight_layout()
    # plt.show()


def get_dls_txts(base_folder: str):
    path_arr = []

    p = Path(base_folder)
    for entry in p.rglob('*'):
        if entry.is_file():
            path = str(entry.absolute())

            if not '.txt' in path:
                continue

            path_arr.append(path)
    return path_arr


def prepare_dataset(base_folder: str):
    filenames = get_dls_txts(base_folder)

    spectra_list = []
    labels_list = []
    nm = None

    for filename in filenames:
        file_name_with_ext = Path(filename).name
        label = int(file_name_with_ext[0])

        labels_list.append(label)
        nm, intensity = read_dls_txt(filename)
        spectra_list.append(intensity)

    return spectra_list, labels_list, nm


def pca_spectroscopy_analysis(spectra_list, labels_list, x, test_size=0.2, random_state=42):
    # Преобразуем в numpy массивы
    X = np.array(spectra_list)  # матрица: [n_samples, n_features]
    y = np.array(labels_list)  # вектор меток

    print(f"Форма данных: {X.shape}")
    print(f"Классы: {np.unique(y)}")

    # 1. Разделение на train/test
    X_train = X
    y_train = y


    # 2. Стандартизация на тренировочных данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit только на train!

    # 3. PCA на стандартизованных тренировочных данных
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)

    # 4. Визуализация результатов
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 4.1 Score plot (PC1 vs PC2) - Train + Test
    unique_labels = np.unique(y)
    colors = ['red', 'blue', 'green', 'orange', 'gray']  # добавьте цвета по необходимости

    for i, label in enumerate(unique_labels):
        # Train points
        train_mask = y_train == label
        axes[0, 0].scatter(X_train_pca[train_mask, 0], X_train_pca[train_mask, 1],
                           c=colors[i], marker='o', alpha=0.7, label=f'{label} (train)')


    axes[0, 0].set_xlabel('PC1 ({:.1f}% дисперсии)'.format(pca.explained_variance_ratio_[0] * 100))
    axes[0, 0].set_ylabel('PC2 ({:.1f}% дисперсии)'.format(pca.explained_variance_ratio_[1] * 100))
    axes[0, 0].set_title('Score Plot: PC1 vs PC2\n(Train = кружки, Test = кресты)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 4.2 Cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    axes[0, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    axes[0, 1].set_xlabel('Количество главных компонент')
    axes[0, 1].set_ylabel('Накопленная дисперсия')
    axes[0, 1].set_title('Накопленная объясненная дисперсия')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% дисперсии')
    axes[0, 1].legend()

    # 4.3 Loadings plot для PC1
    wavelengths = range(X.shape[1])  # замените на реальные длины волн!
    axes[1, 0].plot(x, pca.components_[0, :], '.')
    axes[1, 0].set_xlabel('Длина волны (индекс)')
    axes[1, 0].set_ylabel('Нагрузка')
    axes[1, 0].set_title('Loadings Plot: PC1')
    axes[1, 0].grid(True, alpha=0.3)

    # 4.4 Loadings plot для PC2
    axes[1, 1].plot(x, pca.components_[1, :], '.')
    axes[1, 1].set_xlabel('Длина волны (индекс)')
    axes[1, 1].set_ylabel('Нагрузка')
    axes[1, 1].set_title('Loadings Plot: PC2')
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Вывод дополнительной информации
    print(f"\nОбъясненная дисперсия первых 5 компонент:")
    for i, var in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"PC{i + 1}: {var * 100:.2f}%")

    print(f"Накопленная дисперсия первых 2 компонент: {cumulative_variance[1] * 100:.2f}%")

    plt.tight_layout()
    plt.show()

    return scaler, pca, X_train_pca


base_dir = "C:\\Users\\artur\\Desktop\\Milk PCA feb 26\\дрс\\"
spectra_list, labels_list, x = prepare_dataset(base_dir)
# pca_spectroscopy_analysis(spectra_list, labels_list, x, test_size=0.01)


plt.figure(figsize=(12.8, 4.8))
'''
plot_dls_txt(base_dir + "1\\1 1.txt", 'red')
plot_dls_txt(base_dir + "1\\1 2.txt", 'salmon')
plot_dls_txt(base_dir + "1\\1 3.txt", 'tomato')
plot_dls_txt(base_dir + "1\\1 4.txt", 'brown')
plot_dls_txt(base_dir + "1\\1 5.txt", 'lightcoral')


plot_dls_txt(base_dir + "2\\2 1.txt", 'red')
plot_dls_txt(base_dir + "2\\2 2.txt", 'salmon')
plot_dls_txt(base_dir + "2\\2 4.txt", 'brown')
plot_dls_txt(base_dir + "2\\2 5.txt", 'lightcoral')


plot_dls_txt(base_dir + "3\\3 1.txt", 'red')
plot_dls_txt(base_dir + "3\\3 2.txt", 'salmon')
plot_dls_txt(base_dir + "3\\3 3.txt", 'tomato')
plot_dls_txt(base_dir + "3\\3 4.txt", 'brown')
plot_dls_txt(base_dir + "3\\3 5.txt", 'lightcoral')


plot_dls_txt(base_dir + "4\\4 1.txt", 'red')
plot_dls_txt(base_dir + "4\\4 2.txt", 'salmon')
plot_dls_txt(base_dir + "4\\4 3.txt", 'tomato')
plot_dls_txt(base_dir + "4\\4 4.txt", 'brown')
plot_dls_txt(base_dir + "4\\4 5.txt", 'lightcoral')


plot_dls_txt(base_dir + "5\\5 1.txt", 'red')
plot_dls_txt(base_dir + "5\\5 2.txt", 'salmon')
plot_dls_txt(base_dir + "5\\5 3.txt", 'tomato')
plot_dls_txt(base_dir + "5\\5 4.txt", 'brown')
plot_dls_txt(base_dir + "5\\5 5.txt", 'lightcoral')


plot_dls_txt(base_dir + "4\\4 1.txt", 'cyan')
plot_dls_txt(base_dir + "4\\4 2.txt", 'cadetblue')
plot_dls_txt(base_dir + "4\\4 3.txt", 'skyblue')
plot_dls_txt(base_dir + "4\\4 4.txt", 'deepskyblue')
plot_dls_txt(base_dir + "4\\4 5.txt", 'lightblue')
'''

id = str(5)
plot_dls_txt(base_dir + f"{id}\\{id} 1.txt")
plot_dls_txt(base_dir + f"{id}\\{id} 2.txt")
plot_dls_txt(base_dir + f"{id}\\{id} 3.txt")
plot_dls_txt(base_dir + f"{id}\\{id} 4.txt")
plot_dls_txt(base_dir + f"{id}\\{id} 5.txt")

plt.legend()
plt.show()