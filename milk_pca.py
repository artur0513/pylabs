import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


dir = r'C:\Users\artur\Desktop\Экспериментальные данные\MILK\milk ftir dataset\\'


def read_file(filename: str) -> [np.array, np.array]:
    file = open(filename)
    x = []
    y = []
    counter = 0

    # Read file
    for line in file:
        if line[0].isnumeric() or line[0] == '-':
            values = line.split('\t')
            #if (1370 < float(values[0]) < 1470 or 1120 < float(values[0]) < 1170 or 1690 < float(values[0]) < 1770) and counter % 2 == 0:
            x.append(float(values[0]))
            y.append(float(values[1]))
            counter += 1

    return np.array(x), np.array(y)


def get_base_y():
    _, y_base0 = read_file(dir + 'окно 0.dat')
    _, y_base1 = read_file(dir + 'окно 1.dat')
    _, y_base2 = read_file(dir + 'окно 2.dat')
    return (y_base0 + y_base1 + y_base2) / 3.0


def filter_noise(y):
    n = 4  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    return filtfilt(b, a, y)


def area_under_curve(y: np.array):
    return y.sum()


def specter_pipeline(y_base, filename):
    x, y = read_file(filename)
    y /= y_base
    y = filter_noise(y)

    #absorbed = 1.0 - y
    #sum_absorbed = area_under_curve(absorbed)
    #absorbed /= sum_absorbed
    #absorbed /= np.max(absorbed)
    # y = 1.0 - absorbed

    y = -np.log(y)
    #y /= np.quantile(y, 0.9)
    y = (y - np.mean(y))/np.std(y)
    return x, y


def get_milk_type(filename):
    basename_without_ext = os.path.splitext(os.path.basename(filename))[0]
    if 'см1' in basename_without_ext:
        return 'см1'
    elif 'см2' in basename_without_ext:
        return 'см2'
    elif 'обр0' in basename_without_ext:
        return 'обр0'
    elif 'обр1' in basename_without_ext:
        return 'обр1'
    elif 'обр2' in basename_without_ext:
        return 'обр2'
    else:
        return 'null'


def plot_specter(y_base, filename):
    x, y = specter_pipeline(y_base, filename)
    basename_without_ext = os.path.splitext(os.path.basename(filename))[0]
    plt.plot(x, y, label=basename_without_ext)


def prepare_data_lists():
    y_base = get_base_y()
    spectra_list = []
    labels_list = []
    x = None
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            path = dir + '\\' + file
            milk_type = get_milk_type(path)
            if not 'null' in milk_type:
                x, y = specter_pipeline(y_base, path)
                spectra_list.append(y)
                labels_list.append(milk_type)

    return spectra_list, labels_list, x


def pca_spectroscopy_analysis(spectra_list, labels_list, x, test_size=0.2, random_state=42):
    # Преобразуем в numpy массивы
    X = np.array(spectra_list)  # матрица: [n_samples, n_features]
    y = np.array(labels_list)  # вектор меток

    print(f"Форма данных: {X.shape}")
    print(f"Классы: {np.unique(y)}")

    # 1. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 2. Стандартизация на тренировочных данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit только на train!
    X_test_scaled = scaler.transform(X_test)  # transform на test

    # 3. PCA на стандартизованных тренировочных данных
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # 4. Визуализация результатов
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 4.1 Score plot (PC1 vs PC2) - Train + Test
    unique_labels = np.unique(y)
    colors = ['red', 'blue', 'green', 'orange']  # добавьте цвета по необходимости

    for i, label in enumerate(unique_labels):
        # Train points
        train_mask = y_train == label
        axes[0, 0].scatter(X_train_pca[train_mask, 0], X_train_pca[train_mask, 1],
                           c=colors[i], marker='o', alpha=0.7, label=f'{label} (train)')
        # Test points
        test_mask = y_test == label
        axes[0, 0].scatter(X_test_pca[test_mask, 0], X_test_pca[test_mask, 1],
                           c=colors[i], marker='x', s=80, label=f'{label} (test)')

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

    return scaler, pca, X_train_pca, X_test_pca


pca_spectroscopy_analysis(*prepare_data_lists(), test_size=0.1, random_state=40)

y_base = get_base_y()
plot_specter(y_base, dir + 'см1_1.dat')


plot_specter(y_base, dir + 'см1_12.dat')
plot_specter(y_base, dir + 'см2_1.dat')
plot_specter(y_base, dir + 'см2_11.dat')

plt.legend()
plt.show()
