# =============================================================================
#  matrix.py  —  Pure-Python Matrix Math Engine
#  Every single mathematical primitive used by the transformer lives here.
#
#  Naming convention:
#    A, B, C  = 2-D matrices  (list-of-lists, row-major)
#    x, y, z  = 1-D vectors   (flat list)
#    shape is always written as (rows, cols) or (m, n)
# =============================================================================

import math
import random
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. CREATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def zeros(rows: int, cols: int) -> list:
    """Return an (rows × cols) zero matrix."""
    return [[0.0] * cols for _ in range(rows)]

def zeros_1d(n: int) -> list:
    return [0.0] * n

def ones_1d(n: int) -> list:
    return [1.0] * n

def randn(rows: int, cols: int, std: float = 0.02) -> list:
    return [[random.gauss(0.0, std) for _ in range(cols)] for _ in range(rows)]

def randn_1d(n: int, std: float = 0.02) -> list:
    return [random.gauss(0.0, std) for _ in range(n)]

def copy_2d(A: list) -> list:
    return [row[:] for row in A]

def copy_1d(x: list) -> list:
    return x[:]


# ─────────────────────────────────────────────────────────────────────────────
# 2. CORE MATRIX OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def matmul(A: list, B: list) -> list:
    """Matrix multiply: (m, k) × (k, n)  →  (m, n)  — NumPy accelerated."""
    return np.matmul(
        np.array(A, dtype=np.float64),
        np.array(B, dtype=np.float64)
    ).tolist()

def transpose(A: list) -> list:
    return np.array(A).T.tolist()

def add_2d(A: list, B: list) -> list:
    return (np.array(A) + np.array(B)).tolist()

def sub_2d(A: list, B: list) -> list:
    m, n = len(A), len(A[0])
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(m)]

def mul_2d(A: list, B: list) -> list:
    """Element-wise multiply (Hadamard product)."""
    m, n = len(A), len(A[0])
    return [[A[i][j] * B[i][j] for j in range(n)] for i in range(m)]

def scale_2d(A: list, s: float) -> list:
    return (np.array(A) * s).tolist()
    
def add_bias(A: list, b: list) -> list:
    """Broadcast-add a 1-D bias vector to every row of A."""
    return [[A[i][j] + b[j] for j in range(len(A[0]))] for i in range(len(A))]

def add_1d(a: list, b: list) -> list:
    return [a[i] + b[i] for i in range(len(a))]

def sub_1d(a: list, b: list) -> list:
    return [a[i] - b[i] for i in range(len(a))]

def scale_1d(a: list, s: float) -> list:
    return [v * s for v in a]

def mul_1d(a: list, b: list) -> list:
    return [a[i] * b[i] for i in range(len(a))]

def sum_cols(A: list) -> list:
    """Reduce over rows → column-sum vector of shape (n,)."""
    n = len(A[0])
    result = zeros_1d(n)
    for row in A:
        for j in range(n):
            result[j] += row[j]
    return result

def outer(a: list, b: list) -> list:
    """Outer product: (m,) × (n,)  →  (m, n)"""
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]

def norm_1d(a: list) -> float:
    """L2 norm of a 1-D vector."""
    return math.sqrt(sum(v * v for v in a))

def global_l2_norm(matrices: list, vectors: list = None) -> float:
    """Global L2 norm across matrices and vectors — used for gradient clipping."""
    total = 0.0
    for M in matrices:
        for row in M:
            total += sum(v * v for v in row)
    if vectors:
        for v in vectors:
            total += sum(x * x for x in v)
    return math.sqrt(total)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SOFTMAX  (forward + backward)
# ─────────────────────────────────────────────────────────────────────────────

def _softmax_1d(x: list) -> list:
    """Numerically-stable softmax for a single row vector."""
    mx = max(x)
    e = [math.exp(v - mx) for v in x]
    s = sum(e)
    return [v / s for v in e]

def softmax_rows(A: list) -> list:
    """Apply softmax independently to every row of a 2-D matrix."""
    return [_softmax_1d(row) for row in A]

def softmax_grad_1d(dL: list, s: list) -> list:
    """Backward pass through softmax for one row."""
    dot = sum(dL[j] * s[j] for j in range(len(s)))
    return [s[i] * (dL[i] - dot) for i in range(len(s))]

def softmax_grad_rows(dL: list, S: list) -> list:
    """Row-wise softmax backward.  dL, S: (m, n)  →  dx: (m, n)"""
    return [softmax_grad_1d(dL[i], S[i]) for i in range(len(S))]


# ─────────────────────────────────────────────────────────────────────────────
# 4. GELU ACTIVATION  (forward + backward)
# ─────────────────────────────────────────────────────────────────────────────

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_GELU_C         = 0.044715

def _gelu_scalar(x: float) -> float:
    inner = _SQRT_2_OVER_PI * (x + _GELU_C * x ** 3)
    return 0.5 * x * (1.0 + math.tanh(inner))

def _gelu_grad_scalar(x: float) -> float:
    inner   = _SQRT_2_OVER_PI * (x + _GELU_C * x ** 3)
    t       = math.tanh(inner)
    sech2   = 1.0 - t * t
    d_inner = _SQRT_2_OVER_PI * (1.0 + 3.0 * _GELU_C * x ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner

def gelu_2d(A: list) -> list:
    """Apply GELU element-wise to a 2-D matrix."""
    return [[_gelu_scalar(v) for v in row] for row in A]

def gelu_grad_2d(dL: list, A_pre: list) -> list:
    """Element-wise GELU backward."""
    return [[dL[i][j] * _gelu_grad_scalar(A_pre[i][j])
             for j in range(len(A_pre[0]))]
            for i in range(len(A_pre))]


# ─────────────────────────────────────────────────────────────────────────────
# 5. LAYER NORMALISATION  (forward + backward)
# ─────────────────────────────────────────────────────────────────────────────

def layer_norm_1d(x: list, gamma: list, beta: list, eps: float = 1e-5):
    """Layer-norm a single 1-D vector."""
    n     = len(x)
    mean  = sum(x) / n
    var   = sum((v - mean) ** 2 for v in x) / n
    std   = math.sqrt(var + eps)
    x_hat = [(v - mean) / std for v in x]
    y     = [gamma[j] * x_hat[j] + beta[j] for j in range(n)]
    return y, x_hat, mean, std

def layer_norm_2d(X: list, gamma: list, beta: list, eps: float = 1e-5):
    """Apply layer-norm row-by-row."""
    Y, cache = [], []
    for row in X:
        y, x_hat, mean, std = layer_norm_1d(row, gamma, beta, eps)
        Y.append(y)
        cache.append((x_hat, mean, std))
    return Y, cache

def layer_norm_backward(dL_dY: list, cache: list, gamma: list) -> tuple:
    """Backward pass through layer normalisation."""
    seq_len = len(dL_dY)
    n       = len(dL_dY[0])

    dL_dX     = zeros(seq_len, n)
    dL_dgamma = zeros_1d(n)
    dL_dbeta  = zeros_1d(n)

    for i, (x_hat, mean, std) in enumerate(cache):
        dy = dL_dY[i]

        for j in range(n):
            dL_dgamma[j] += dy[j] * x_hat[j]
            dL_dbeta[j]  += dy[j]

        dy_hat       = [dy[j] * gamma[j] for j in range(n)]
        sum_dy_hat   = sum(dy_hat)
        sum_dy_hat_x = sum(dy_hat[j] * x_hat[j] for j in range(n))

        for j in range(n):
            dL_dX[i][j] = (1.0 / (n * std)) * (
                n * dy_hat[j]
                - sum_dy_hat
                - x_hat[j] * sum_dy_hat_x
            )

    return dL_dX, dL_dgamma, dL_dbeta


# ─────────────────────────────────────────────────────────────────────────────
# 6. CAUSAL MASK
# ─────────────────────────────────────────────────────────────────────────────

def causal_mask(seq_len: int) -> list:
    """Upper-triangular mask that blocks attention to future positions."""
    NEG_INF = -1e9
    mask = zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                mask[i][j] = NEG_INF
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 7. MULTI-HEAD SPLIT / MERGE
# ─────────────────────────────────────────────────────────────────────────────

def split_heads(X: list, n_heads: int, d_head: int) -> list:
    """Reshape (seq_len, d_model) into list of n_heads matrices (seq_len, d_head)."""
    seq_len = len(X)
    return [
        [[X[t][h * d_head + j] for j in range(d_head)]
         for t in range(seq_len)]
        for h in range(n_heads)
    ]

def merge_heads(heads: list, n_heads: int, d_head: int) -> list:
    """Inverse of split_heads: list of (seq_len, d_head) → (seq_len, d_model)"""
    seq_len = len(heads[0])
    result  = zeros(seq_len, n_heads * d_head)
    for h, head in enumerate(heads):
        for t in range(seq_len):
            for j in range(d_head):
                result[t][h * d_head + j] = head[t][j]
    return result

def clip_grad_1d(x: list, max_val: float) -> list:
    """Clip a 1-D gradient vector element-wise."""
    return [max(-max_val, min(max_val, v)) for v in x]