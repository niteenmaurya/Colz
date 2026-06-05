# =============================================================================
#  matrix.py  —  NumPy-Accelerated Matrix Engine
#  Sab kuch NumPy se — pure Python loops hataaye gaye
# =============================================================================

import math
import random
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. CREATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def zeros(rows: int, cols: int) -> list:
    return np.zeros((rows, cols), dtype=np.float64).tolist()

def zeros_1d(n: int) -> list:
    return np.zeros(n, dtype=np.float64).tolist()

def ones_1d(n: int) -> list:
    return np.ones(n, dtype=np.float64).tolist()

def randn(rows: int, cols: int, std: float = 0.02) -> list:
    return (np.random.randn(rows, cols) * std).tolist()

def randn_1d(n: int, std: float = 0.02) -> list:
    return (np.random.randn(n) * std).tolist()

def copy_2d(A: list) -> list:
    return np.array(A).tolist()

def copy_1d(x: list) -> list:
    return list(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CORE MATRIX OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def matmul(A: list, B: list) -> list:
    """Matrix multiply: (m, k) × (k, n) → (m, n)"""
    return np.matmul(
        np.array(A, dtype=np.float64),
        np.array(B, dtype=np.float64)
    ).tolist()

def transpose(A: list) -> list:
    return np.array(A, dtype=np.float64).T.tolist()

def add_2d(A: list, B: list) -> list:
    return (np.array(A, dtype=np.float64) +
            np.array(B, dtype=np.float64)).tolist()

def sub_2d(A: list, B: list) -> list:
    return (np.array(A, dtype=np.float64) -
            np.array(B, dtype=np.float64)).tolist()

def mul_2d(A: list, B: list) -> list:
    return (np.array(A, dtype=np.float64) *
            np.array(B, dtype=np.float64)).tolist()

def scale_2d(A: list, s: float) -> list:
    return (np.array(A, dtype=np.float64) * s).tolist()

def add_bias(A: list, b: list) -> list:
    return (np.array(A, dtype=np.float64) +
            np.array(b, dtype=np.float64)).tolist()

def add_1d(a: list, b: list) -> list:
    return (np.array(a) + np.array(b)).tolist()

def sub_1d(a: list, b: list) -> list:
    return (np.array(a) - np.array(b)).tolist()

def scale_1d(a: list, s: float) -> list:
    return (np.array(a) * s).tolist()

def mul_1d(a: list, b: list) -> list:
    return (np.array(a) * np.array(b)).tolist()

def sum_cols(A: list) -> list:
    return np.array(A, dtype=np.float64).sum(axis=0).tolist()

def outer(a: list, b: list) -> list:
    return np.outer(a, b).tolist()

def norm_1d(a: list) -> float:
    return float(np.linalg.norm(a))

def global_l2_norm(matrices: list, vectors: list = None) -> float:
    total = 0.0
    for M in matrices:
        total += np.sum(np.array(M) ** 2)
    if vectors:
        for v in vectors:
            total += np.sum(np.array(v) ** 2)
    return float(np.sqrt(total))


# ─────────────────────────────────────────────────────────────────────────────
# 3. SOFTMAX  (forward + backward)
# ─────────────────────────────────────────────────────────────────────────────

def softmax_rows(A: list) -> list:
    A_np = np.array(A, dtype=np.float64)
    A_np -= A_np.max(axis=1, keepdims=True)   # stable
    e = np.exp(A_np)
    return (e / e.sum(axis=1, keepdims=True)).tolist()

def softmax_grad_1d(dL: list, s: list) -> list:
    dL_np = np.array(dL)
    s_np  = np.array(s)
    dot   = np.dot(dL_np, s_np)
    return (s_np * (dL_np - dot)).tolist()

def softmax_grad_rows(dL: list, S: list) -> list:
    dL_np = np.array(dL, dtype=np.float64)
    S_np  = np.array(S,  dtype=np.float64)
    dot   = (dL_np * S_np).sum(axis=1, keepdims=True)
    return (S_np * (dL_np - dot)).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 4. GELU  (forward + backward)
# ─────────────────────────────────────────────────────────────────────────────

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_GELU_C         = 0.044715

def gelu_2d(A: list) -> list:
    x     = np.array(A, dtype=np.float64)
    inner = _SQRT_2_OVER_PI * (x + _GELU_C * x ** 3)
    return (0.5 * x * (1.0 + np.tanh(inner))).tolist()

def gelu_grad_2d(dL: list, A_pre: list) -> list:
    x       = np.array(A_pre, dtype=np.float64)
    dL_np   = np.array(dL,    dtype=np.float64)
    inner   = _SQRT_2_OVER_PI * (x + _GELU_C * x ** 3)
    t       = np.tanh(inner)
    sech2   = 1.0 - t ** 2
    d_inner = _SQRT_2_OVER_PI * (1.0 + 3.0 * _GELU_C * x ** 2)
    grad    = 0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner
    return (dL_np * grad).tolist()

# scalar helpers (used internally — keep for compatibility)
def _gelu_scalar(x):
    inner = _SQRT_2_OVER_PI * (x + _GELU_C * x**3)
    return 0.5 * x * (1.0 + math.tanh(inner))

def _gelu_grad_scalar(x):
    inner   = _SQRT_2_OVER_PI * (x + _GELU_C * x**3)
    t       = math.tanh(inner)
    d_inner = _SQRT_2_OVER_PI * (1.0 + 3.0 * _GELU_C * x**2)
    return 0.5*(1+t) + 0.5*x*(1-t*t)*d_inner


# ─────────────────────────────────────────────────────────────────────────────
# 5. LAYER NORMALISATION  (forward + backward)
# ─────────────────────────────────────────────────────────────────────────────

def layer_norm_1d(x: list, gamma: list, beta: list, eps: float = 1e-5):
    x_np  = np.array(x,     dtype=np.float64)
    g_np  = np.array(gamma, dtype=np.float64)
    b_np  = np.array(beta,  dtype=np.float64)
    mean  = x_np.mean()
    std   = math.sqrt(x_np.var() + eps)
    x_hat = ((x_np - mean) / std).tolist()
    y     = (g_np * np.array(x_hat) + b_np).tolist()
    return y, x_hat, float(mean), float(std)

def layer_norm_2d(X: list, gamma: list, beta: list, eps: float = 1e-5):
    Y, cache = [], []
    for row in X:
        y, x_hat, mean, std = layer_norm_1d(row, gamma, beta, eps)
        Y.append(y)
        cache.append((x_hat, mean, std))
    return Y, cache

def layer_norm_backward(dL_dY: list, cache: list, gamma: list) -> tuple:
    dL_np = np.array(dL_dY, dtype=np.float64)   # (seq, n)
    g_np  = np.array(gamma, dtype=np.float64)    # (n,)
    seq_len, n = dL_np.shape

    dL_dX     = np.zeros_like(dL_np)
    dL_dgamma = np.zeros(n)
    dL_dbeta  = np.zeros(n)

    for i, (x_hat, mean, std) in enumerate(cache):
        xh  = np.array(x_hat)
        dy  = dL_np[i]
        dL_dgamma += dy * xh
        dL_dbeta  += dy
        dy_hat       = dy * g_np
        sum_dy_hat   = dy_hat.sum()
        sum_dy_hat_x = (dy_hat * xh).sum()
        dL_dX[i] = (1.0 / (n * std)) * (
            n * dy_hat - sum_dy_hat - xh * sum_dy_hat_x
        )

    return dL_dX.tolist(), dL_dgamma.tolist(), dL_dbeta.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 6. CAUSAL MASK
# ─────────────────────────────────────────────────────────────────────────────

def causal_mask(seq_len: int) -> list:
    mask = np.zeros((seq_len, seq_len), dtype=np.float64)
    mask[np.triu_indices(seq_len, k=1)] = -1e9
    return mask.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 7. MULTI-HEAD SPLIT / MERGE
# ─────────────────────────────────────────────────────────────────────────────

def split_heads(X: list, n_heads: int, d_head: int) -> list:
    X_np    = np.array(X, dtype=np.float64)          # (seq, d_model)
    seq_len = X_np.shape[0]
    X_np    = X_np.reshape(seq_len, n_heads, d_head)  # (seq, h, d_head)
    X_np    = X_np.transpose(1, 0, 2)                 # (h, seq, d_head)
    return [X_np[h].tolist() for h in range(n_heads)]

def merge_heads(heads: list, n_heads: int, d_head: int) -> list:
    # heads: list of (seq, d_head)
    arr = np.stack([np.array(h) for h in heads], axis=0)  # (h, seq, d_head)
    arr = arr.transpose(1, 0, 2)                           # (seq, h, d_head)
    seq_len = arr.shape[0]
    return arr.reshape(seq_len, n_heads * d_head).tolist()

def clip_grad_1d(x: list, max_val: float) -> list:
    return np.clip(np.array(x), -max_val, max_val).tolist()