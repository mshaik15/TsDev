# 📊 Time Series to Vector Pipeline

Given a time series:

$$
X = \begin{bmatrix} x_1 \\\\ x_2 \\\\ \vdots \\\\ x_T \end{bmatrix} \in \mathbb{R}^T
$$

---

## Step 1: Sliding Window Extraction

For each window \( i = 1, 2, \dots, N \) where

$$
N = \left\lfloor \frac{T - w}{s} \right\rfloor + 1
$$

with window size \( w \) and stride \( s \), extract the window:

$$
W_i = \begin{bmatrix} x_{i} \\\\ x_{i+1} \\\\ \vdots \\\\ x_{i+w-1} \end{bmatrix} \in \mathbb{R}^w
$$

---

## Step 2: Feature Extraction per Window

**(a) Statistical feature vector:**

$$
v_i^{(\text{stat})} = \begin{bmatrix}
\mu_i \\\\ \sigma_i^2 \\\\ \gamma_i \\\\ \kappa_i \\\\ \min_i \\\\ \max_i \\\\ \mathrm{median}_i \\\\ \vdots
\end{bmatrix}
\in \mathbb{R}^{d_s}
$$

Each scalar is computed from \( W_i \).

**(b) Frequency (FFT) feature vector:**

Compute DFT:

$\hat{X}_i[k] = \sum_{n=0}^{w-1} (W_i)_n \cdot e^{-j \frac{2\pi k n}{w}}, \quad k = 0, \dots, w-1$X

Compute magnitude:

$$
|\hat{X}_i[k]| = \sqrt{\mathrm{Re}(\hat{X}_i[k])^2 + \mathrm{Im}(\hat{X}_i[k])^2}
$$

Take first \( K \) non-DC components:

$$
v_i^{(\text{fft})} = \begin{bmatrix}
|\hat{X}_i[1]| \\\\ |\hat{X}_i[2]| \\\\ \vdots \\\\ |\hat{X}_i[K]|
\end{bmatrix}
\in \mathbb{R}^{d_f}, \quad d_f = K
$$

---

## Step 3: Concatenate Final Feature Vector

$$
v_i = \begin{bmatrix}
v_i^{(\text{stat})} \\\\ v_i^{(\text{fft})}
\end{bmatrix}
\in \mathbb{R}^{d_s + d_f}
$$

---

## Step 4: Construct Final Feature Matrix

Stack vectors horizontally:

$$
V = \begin{bmatrix}
| & | & & | \\\\
v_1 & v_2 & \cdots & v_N \\\\
| & | & & |
\end{bmatrix}
\in \mathbb{R}^{(d_s + d_f) \times N}
$$

---

## ✅ Summary

- \( V \) has **columns**: one per window.
- \( V \) has **rows**: one per feature dimension.
- Ready for **storage** or **ML model input**.
