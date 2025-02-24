import os
import scipy.io
from scipy.stats import skew, kurtosis
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import ttest_ind
from scipy.signal import hilbert
from antropy import spectral_entropy

# Function to load data from a MATLAB file
def load_data(file_path):
    mat_data = loadmat(file_path)
    if 'signal' in mat_data:
        data = mat_data['signal'][0,:].squeeze()
    else:
        raise ValueError(f"Data variable not found in {file_path}.")
    return data

# Function to calculate features using a sliding window
def calculate_features_with_sliding_window(data, window_size, step_size):
    num_windows = (len(data) - window_size) // step_size + 1
    features = {
        'Mean': [],
        'Standard Deviation': [],
        'Skewness': [],
        'Kurtosis': [],
        'RMS': [],
        'Peak-to-Peak': [],
        'Interquartile Range (IQR)': [],
        'Peak Count': [],
        'Zero-Crossing Rate': [],
        'Rank-Based Entropy': [],
        'Fractal Geometry Indicator': [],
        'Chaos Quantifier': []
    }

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        window_data = data[start:end]

        features['Mean'].append(np.mean(window_data))
        features['Standard Deviation'].append(np.std(window_data))
        features['Skewness'].append(skew(window_data))
        features['Kurtosis'].append(kurtosis(window_data))
        features['RMS'].append(np.sqrt(np.mean(np.square(window_data))))
        features['Peak-to-Peak'].append(np.ptp(window_data))
        q75, q25 = np.percentile(window_data, [75, 25])
        features['Interquartile Range (IQR)'].append(q75 - q25)
        peak_count = len(np.where(np.diff(np.sign(window_data)))[0])
        features['Peak Count'].append(peak_count)
        zero_crossings = np.sum(np.diff(np.sign(window_data)) != 0)
        features['Zero-Crossing Rate'].append(zero_crossings / window_size)
        # Entropy-Based Measures
        features['Rank-Based Entropy'].append(rb_entropy(window_data, order=3, normalize=True))
        # Fractal Geometry Indicator (using Higuchi's method)
        features['Fractal Geometry Indicator'].append(higuchi_fd(window_data, kmax=5))
        # Chaos Quantifier
        features['Chaos Quantifier'].append(chao(window_data))

    return features

# Function to calculate the Fractal Geometry Indicator
def higuchi_fd(x, kmax):
    L = []
    x_len = len(x)
    for k in range(1, kmax+1):
        Lk = []
        for m in range(k):
            Lmk = np.sum(np.abs(np.diff(x[m::k])) * (x_len - 1) / ((x_len - m) // k * k))
            Lk.append(Lmk)
        L.append(np.mean(Lk))
    L = np.array(L)
    reg = np.polyfit(np.log(range(1, kmax + 1)), np.log(L), 1)
    return reg[0]

# Function to estimate the Chaos Quantifier
def chao(time_series):
    N = len(time_series)
    x_t = time_series[1:]
    x_t_1 = time_series[:-1]
    delta_x = np.abs(x_t - x_t_1)
    lyap_exp = np.mean(np.log(delta_x + 1e-10))
    return lyap_exp


