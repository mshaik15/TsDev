import numpy as np
from scipy.fft import fft
from scipy.stats import skew, kurtosis

# Step 1: Sliding Window Extraction
def extract_sliding_windows(x: np.ndarray, w: int, s: int) -> np.ndarray:
    T = len(x)
    N = (T - w) // s + 1
    return np.array([x[i * s : i * s + w] for i in range(N)])

# Step 2a: Statistical Feature Extraction
def compute_statistical_features(window: np.ndarray) -> np.ndarray:
    mu = np.mean(window)
    sigma2 = np.var(window)
    gamma = skew(window)
    kappa = kurtosis(window)
    min_val = np.min(window)
    max_val = np.max(window)
    median = np.median(window)
    return np.array([mu, sigma2, gamma, kappa, min_val, max_val, median])

# Step 2b: Frequency Feature Extraction (FFT Magnitudes)
def compute_fft_features(window: np.ndarray, K: int) -> np.ndarray:
    fft_coeffs = fft(window)
    magnitudes = np.abs(fft_coeffs)[1:K + 1]  # Exclude DC component at index 0
    return magnitudes

# Step 3: Concatenate Statistical + FFT Features
def extract_feature_vector(window: np.ndarray, K: int) -> np.ndarray:
    stat_feats = compute_statistical_features(window)
    fft_feats = compute_fft_features(window, K)
    return np.concatenate([stat_feats, fft_feats])

# Step 4: Full Feature Matrix V
def build_feature_matrix(x: np.ndarray, w: int, s: int, K: int) -> np.ndarray:
    windows = extract_sliding_windows(x, w, s)
    feature_matrix = np.array([extract_feature_vector(win, K) for win in windows])
    return feature_matrix.T  # Transpose to match LaTeX: columns = window features