# =============================================================================
#  transformer.py  —  GPT-Style Decoder-Only Transformer
#
#  Contains every building block, from the smallest linear layer up to
#  the full GPT model.  Each class exposes:
#      forward(x)   → output
#      backward(dL) → gradient w.r.t. input (+ accumulates weight grads)
#      zero_grad()  → reset all gradient accumulators
#      params_and_grads() → list of (param, grad, name) for the optimiser
#
#  Architecture:
#    Input IDs
#       ↓
#    Embedding + Positional Encoding          (embeddings.py)
#       ↓
#    TransformerBlock × n_layers              (this file)
#       ├─ LayerNorm → MultiHeadAttention → residual
#       └─ LayerNorm → FeedForward         → residual
#       ↓
#    LayerNorm  (final)
#       ↓
#    Linear  (LM head: d_model → vocab_size)
#       ↓
#    Logits  (seq_len, vocab_size)
# =============================================================================

import math
from matrix import (
    zeros, zeros_1d, ones_1d, randn, randn_1d,
    matmul, transpose, add_2d, add_bias, scale_2d,
    mul_2d, sum_cols,
    softmax_rows, softmax_grad_rows,
    gelu_2d, gelu_grad_2d,
    layer_norm_2d, layer_norm_backward,
    causal_mask, split_heads, merge_heads,
)
from embeddings import Embedding


# ─────────────────────────────────────────────────────────────────────────────
# 1. LINEAR LAYER
# ─────────────────────────────────────────────────────────────────────────────

class Linear:
    """
    Y = X @ W + b

    X : (seq_len, in_dim)
    W : (in_dim, out_dim)   — learnable
    b : (out_dim,)          — learnable bias
    Y : (seq_len, out_dim)

    Backward:
        dX = dY @ Wᵀ
        dW = Xᵀ @ dY
        db = Σ dY  (sum over seq positions)
    """

    def __init__(self, in_dim: int, out_dim: int, std: float = 0.02) -> None:
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.W  = randn(in_dim, out_dim, std=std)
        self.b  = zeros_1d(out_dim)
        self.dW = zeros(in_dim, out_dim)
        self.db = zeros_1d(out_dim)
        self._X = None          # cache for backward

    def forward(self, X: list) -> list:
        self._X = X
        return add_bias(matmul(X, self.W), self.b)

    def backward(self, dL: list) -> list:
        """Returns dX (gradient w.r.t. input)."""
        X = self._X
        # dW = Xᵀ @ dL
        XT = transpose(X)
        dW_new = matmul(XT, dL)
        for i in range(self.in_dim):
            for j in range(self.out_dim):
                self.dW[i][j] += dW_new[i][j]
        # db = sum over seq positions
        db_new = sum_cols(dL)
        for j in range(self.out_dim):
            self.db[j] += db_new[j]
        # dX = dL @ Wᵀ
        return matmul(dL, transpose(self.W))

    def zero_grad(self) -> None:
        self.dW = zeros(self.in_dim, self.out_dim)
        self.db = zeros_1d(self.out_dim)

    def params_and_grads(self) -> list:
        return [
            (self.W, self.dW, "linear.W"),
            (self.b, self.db, "linear.b"),   # 1-D
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. LAYER NORM WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class LayerNorm:
    """
    Thin wrapper around the matrix.py layer_norm functions.
    Holds learnable gamma (scale) and beta (shift) vectors.
    """

    def __init__(self, d_model: int) -> None:
        self.d_model = d_model
        self.gamma  = ones_1d(d_model)   # init to 1 (identity scaling)
        self.beta   = zeros_1d(d_model)  # init to 0 (no shift)
        self.dgamma = zeros_1d(d_model)
        self.dbeta  = zeros_1d(d_model)
        self._cache = None

    def forward(self, X: list) -> list:
        Y, cache = layer_norm_2d(X, self.gamma, self.beta)
        self._cache = cache
        return Y

    def backward(self, dL: list) -> list:
        dX, dg, db = layer_norm_backward(dL, self._cache, self.gamma)
        for j in range(self.d_model):
            self.dgamma[j] += dg[j]
            self.dbeta[j]  += db[j]
        return dX

    def zero_grad(self) -> None:
        self.dgamma = zeros_1d(self.d_model)
        self.dbeta  = zeros_1d(self.d_model)

    def params_and_grads(self) -> list:
        return [
            (self.gamma, self.dgamma, "ln.gamma"),
            (self.beta,  self.dbeta,  "ln.beta"),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. MULTI-HEAD SELF-ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention:
    """
    Scaled Dot-Product Multi-Head Self-Attention with causal mask.

    For n_heads = h and d_model = D:
        d_head = D // h

    Forward
    -------
        Q  = X @ Wq        →  (seq, D)
        K  = X @ Wk        →  (seq, D)
        V  = X @ Wv        →  (seq, D)

        Split Q,K,V into h slices along the last dim → each (seq, d_head)

        For head i:
            scores_i  = Q_i @ K_iᵀ / √d_head   →  (seq, seq)
            scores_i += causal_mask              (future positions = −10⁹)
            A_i       = softmax_rows(scores_i)   →  (seq, seq)
            out_i     = A_i @ V_i                →  (seq, d_head)

        merged = concat(out_0, …, out_{h-1})     →  (seq, D)
        output = merged @ Wo                     →  (seq, D)

    Backward
    --------
    Chain rule through every step above in reverse order.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        std = (d_model) ** -0.5     # common initialisation

        # Projection weights (in_dim = d_model, out_dim = d_model)
        self.Wq = randn(d_model, d_model, std=std)
        self.Wk = randn(d_model, d_model, std=std)
        self.Wv = randn(d_model, d_model, std=std)
        self.Wo = randn(d_model, d_model, std=std)

        # Gradient accumulators (same shape as weights)
        self.dWq = zeros(d_model, d_model)
        self.dWk = zeros(d_model, d_model)
        self.dWv = zeros(d_model, d_model)
        self.dWo = zeros(d_model, d_model)

        self._cache = None

    def forward(self, X: list) -> list:
        seq_len   = len(X)
        scale_val = 1.0 / math.sqrt(self.d_head)
        mask      = causal_mask(seq_len)

        # ── Linear projections ────────────────────────────────────────────────
        Q = matmul(X, self.Wq)    # (seq, d_model)
        K = matmul(X, self.Wk)
        V = matmul(X, self.Wv)

        # ── Split into heads ─────────────────────────────────────────────────
        Q_h = split_heads(Q, self.n_heads, self.d_head)   # list of (seq, d_head)
        K_h = split_heads(K, self.n_heads, self.d_head)
        V_h = split_heads(V, self.n_heads, self.d_head)

        head_outs = []
        attn_list = []   # attention weight matrices (one per head)

        for q, k, v in zip(Q_h, K_h, V_h):
            # Scores: QKᵀ / √d_head  +  causal mask
            raw = matmul(q, transpose(k))            # (seq, seq)
            raw = scale_2d(raw, scale_val)
            raw = add_2d(raw, mask)                  # mask future tokens

            # Attention weights + weighted values
            A   = softmax_rows(raw)                  # (seq, seq)
            out = matmul(A, v)                       # (seq, d_head)

            head_outs.append(out)
            attn_list.append(A)

        # ── Merge heads + output projection ──────────────────────────────────
        merged = merge_heads(head_outs, self.n_heads, self.d_head)  # (seq, d_model)
        output = matmul(merged, self.Wo)                             # (seq, d_model)

        # ── Cache everything needed for backward ─────────────────────────────
        self._cache = (X, Q, K, V, Q_h, K_h, V_h, head_outs, attn_list, merged, scale_val)

        return output

    def backward(self, dout: list) -> list:
        """
        Backpropagate through multi-head attention.
        Returns dX (gradient w.r.t. the input X).
        """
        (X, Q, K, V,
         Q_h, K_h, V_h,
         head_outs, attn_list,
         merged, scale_val) = self._cache

        # ── Backprop through output projection Wo ─────────────────────────────
        # output = merged @ Wo
        # dWo    = mergedᵀ @ dout
        # dmerged = dout @ Woᵀ
        dWo_new = matmul(transpose(merged), dout)
        for i in range(self.d_model):
            for j in range(self.d_model):
                self.dWo[i][j] += dWo_new[i][j]

        dmerged = matmul(dout, transpose(self.Wo))     # (seq, d_model)

        # ── Split dmerged back into per-head gradients ───────────────────────
        d_head_outs = split_heads(dmerged, self.n_heads, self.d_head)

        dQ_h, dK_h, dV_h = [], [], []

        for h_idx in range(self.n_heads):
            dout_h = d_head_outs[h_idx]      # (seq, d_head)
            A      = attn_list[h_idx]        # (seq, seq)
            v      = V_h[h_idx]             # (seq, d_head)
            q      = Q_h[h_idx]
            k      = K_h[h_idx]

            # ── Backward through out = A @ V ─────────────────────────────────
            # dA = dout_h @ Vᵀ        (seq, seq)
            # dV = Aᵀ  @ dout_h       (seq, d_head)
            dA = matmul(dout_h, transpose(v))
            dv = matmul(transpose(A), dout_h)

            # ── Backward through A = softmax(scores) ─────────────────────────
            dscores = softmax_grad_rows(dA, A)           # (seq, seq)

            # ── Backward through scores = Q @ Kᵀ * scale ────────────────────
            # dQ = dscores @ K * scale
            # dK = dscoresᵀ @ Q * scale
            dq = scale_2d(matmul(dscores,             k),  scale_val)
            dk = scale_2d(matmul(transpose(dscores),  q),  scale_val)

            dQ_h.append(dq)
            dK_h.append(dk)
            dV_h.append(dv)

        # ── Merge per-head gradients ─────────────────────────────────────────
        dQ = merge_heads(dQ_h, self.n_heads, self.d_head)   # (seq, d_model)
        dK = merge_heads(dK_h, self.n_heads, self.d_head)
        dV = merge_heads(dV_h, self.n_heads, self.d_head)

        # ── Backprop through Q=X@Wq, K=X@Wk, V=X@Wv ─────────────────────────
        XT = transpose(X)
        for mat, dW_acc in [(dQ, self.dWq), (dK, self.dWk), (dV, self.dWv)]:
            new_dW = matmul(XT, mat)
            for i in range(self.d_model):
                for j in range(self.d_model):
                    dW_acc[i][j] += new_dW[i][j]

        dX = matmul(dQ, transpose(self.Wq))
        dX = add_2d(dX, matmul(dK, transpose(self.Wk)))
        dX = add_2d(dX, matmul(dV, transpose(self.Wv)))

        return dX

    def zero_grad(self) -> None:
        self.dWq = zeros(self.d_model, self.d_model)
        self.dWk = zeros(self.d_model, self.d_model)
        self.dWv = zeros(self.d_model, self.d_model)
        self.dWo = zeros(self.d_model, self.d_model)

    def params_and_grads(self) -> list:
        return [
            (self.Wq, self.dWq, "attn.Wq"),
            (self.Wk, self.dWk, "attn.Wk"),
            (self.Wv, self.dWv, "attn.Wv"),
            (self.Wo, self.dWo, "attn.Wo"),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEED-FORWARD NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward:
    """
    Position-wise Feed-Forward Network:

        hidden = GELU( X @ W1 + b1 )      X: (seq, d_model)  →  (seq, d_ff)
        output =        hidden @ W2 + b2   →  (seq, d_model)

    A two-layer MLP applied identically to every token position.
    d_ff is typically 4 × d_model (GPT convention).
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        self.d_model = d_model
        self.d_ff    = d_ff

        self.W1 = randn(d_model, d_ff,    std=0.02)
        self.b1 = zeros_1d(d_ff)
        self.W2 = randn(d_ff,    d_model, std=0.02)
        self.b2 = zeros_1d(d_model)

        self.dW1 = zeros(d_model, d_ff)
        self.db1 = zeros_1d(d_ff)
        self.dW2 = zeros(d_ff, d_model)
        self.db2 = zeros_1d(d_model)

        # Cached for backward
        self._X    = None
        self._pre  = None   # X @ W1 + b1  (pre-GELU)
        self._h    = None   # GELU(pre)

    def forward(self, X: list) -> list:
        pre    = add_bias(matmul(X, self.W1), self.b1)   # (seq, d_ff)
        hidden = gelu_2d(pre)                            # (seq, d_ff)
        output = add_bias(matmul(hidden, self.W2), self.b2)  # (seq, d_model)
        self._X   = X
        self._pre = pre
        self._h   = hidden
        return output

    def backward(self, dout: list) -> list:
        """Returns dX (gradient w.r.t. the FF input)."""
        X, pre, h = self._X, self._pre, self._h

        # ── W2 backward ───────────────────────────────────────────────────────
        dW2_new = matmul(transpose(h), dout)
        for i in range(self.d_ff):
            for j in range(self.d_model):
                self.dW2[i][j] += dW2_new[i][j]
        db2_new = sum_cols(dout)
        for j in range(self.d_model):
            self.db2[j] += db2_new[j]

        # ── GELU backward ─────────────────────────────────────────────────────
        dh   = matmul(dout, transpose(self.W2))   # (seq, d_ff)
        dpre = gelu_grad_2d(dh, pre)              # element-wise chain rule

        # ── W1 backward ───────────────────────────────────────────────────────
        dW1_new = matmul(transpose(X), dpre)
        for i in range(self.d_model):
            for j in range(self.d_ff):
                self.dW1[i][j] += dW1_new[i][j]
        db1_new = sum_cols(dpre)
        for j in range(self.d_ff):
            self.db1[j] += db1_new[j]

        dX = matmul(dpre, transpose(self.W1))     # (seq, d_model)
        return dX

    def zero_grad(self) -> None:
        self.dW1 = zeros(self.d_model, self.d_ff)
        self.db1 = zeros_1d(self.d_ff)
        self.dW2 = zeros(self.d_ff, self.d_model)
        self.db2 = zeros_1d(self.d_model)

    def params_and_grads(self) -> list:
        return [
            (self.W1, self.dW1, "ff.W1"),
            (self.b1, self.db1, "ff.b1"),
            (self.W2, self.dW2, "ff.W2"),
            (self.b2, self.db2, "ff.b2"),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRANSFORMER BLOCK  (Pre-LN variant)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock:
    """
    One transformer decoder layer.

    Pre-LayerNorm layout (more stable training than post-LN):

        x = x + Attention( LN1(x) )
        x = x + FFN(      LN2(x) )

    The '+' operations are residual connections — they let gradients
    flow directly from output to input without passing through attention/FFN,
    preventing vanishing gradients in deep models.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff   = FeedForward(d_model, d_ff)
        self.ln1  = LayerNorm(d_model)
        self.ln2  = LayerNorm(d_model)

        # Caches filled during forward
        self._x      = None
        self._x_ln1  = None
        self._x1     = None   # after first residual
        self._x_ln2  = None

    def forward(self, x: list) -> list:
        # ── Attention sub-layer ───────────────────────────────────────────────
        x_ln1 = self.ln1.forward(x)          # (seq, d_model)
        attn_out = self.attn.forward(x_ln1)  # (seq, d_model)
        x1 = add_2d(x, attn_out)             # residual +

        # ── Feed-Forward sub-layer ────────────────────────────────────────────
        x_ln2 = self.ln2.forward(x1)         # (seq, d_model)
        ff_out = self.ff.forward(x_ln2)      # (seq, d_model)
        x2 = add_2d(x1, ff_out)              # residual +

        # Cache for backward
        self._x     = x
        self._x_ln1 = x_ln1
        self._x1    = x1
        self._x_ln2 = x_ln2

        return x2

    def backward(self, dx2: list) -> list:
        """
        Backpropagate through the block.
        dx2: gradient w.r.t. the block output (seq, d_model)
        Returns: gradient w.r.t. the block input x
        """
        # ── FFN sub-layer backward ────────────────────────────────────────────
        # x2 = x1 + ff_out  →  dx1_ff = dx2;  dff_out = dx2
        dx1  = dx2                           # gradient through residual shortcut
        dff  = dx2                           # gradient through FFN branch
        dx_ln2 = self.ff.backward(dff)       # (seq, d_model)
        dx1  = add_2d(dx1, self.ln2.backward(dx_ln2))   # accumulate from LN2

        # ── Attention sub-layer backward ──────────────────────────────────────
        # x1 = x + attn_out  →  dx_res = dx1;  dattn_out = dx1
        dx   = dx1                                       # gradient through residual shortcut
        dattn = dx1
        dx_ln1 = self.attn.backward(dattn)              # (seq, d_model)
        dx   = add_2d(dx, self.ln1.backward(dx_ln1))   # accumulate from LN1

        return dx

    def zero_grad(self) -> None:
        self.attn.zero_grad()
        self.ff.zero_grad()
        self.ln1.zero_grad()
        self.ln2.zero_grad()

    def params_and_grads(self) -> list:
        params = []
        for sub in [self.attn, self.ff, self.ln1, self.ln2]:
            params.extend(sub.params_and_grads())
        return params


# ─────────────────────────────────────────────────────────────────────────────
# 6. GPT MODEL  (full decoder-only transformer)
# ─────────────────────────────────────────────────────────────────────────────

class GPT:
    """
    A mini GPT-style language model built entirely from scratch.

    Forward pass:
        token_ids  →  Embedding  →  N × TransformerBlock  →
        LayerNorm  →  LM head (Linear)  →  logits

    Logits shape: (seq_len, vocab_size)
    Each logit[i] is the unnormalised probability distribution over the
    next token, given tokens 0…i.

    Config
    ------
    vocab_size : int   — from the tokenizer
    d_model    : int   — embedding dimension (e.g. 64)
    n_heads    : int   — number of attention heads (e.g. 4)
    n_layers   : int   — number of transformer blocks (e.g. 2)
    d_ff       : int   — FFN hidden dim, typically 4 × d_model
    max_seq_len: int   — maximum context window length
    """

    def __init__(
        self,
        vocab_size:  int,
        d_model:     int,
        n_heads:     int,
        n_layers:    int,
        d_ff:        int,
        max_seq_len: int,
    ) -> None:
        self.vocab_size  = vocab_size
        self.d_model     = d_model

        # ── Layers ────────────────────────────────────────────────────────────
        self.embedding = Embedding(vocab_size, d_model, max_seq_len)
        self.blocks    = [TransformerBlock(d_model, n_heads, d_ff)
                          for _ in range(n_layers)]
        self.ln_final  = LayerNorm(d_model)
        self.lm_head   = Linear(d_model, vocab_size, std=0.02)

        # Forward-pass cache
        self._x_after_blocks = None

    def forward(self, token_ids: list) -> list:
        """
        Args:
            token_ids : list of int, shape (seq_len,)

        Returns:
            logits : (seq_len, vocab_size)  — raw un-normalised scores
        """
        # Token + Positional embeddings
        x = self.embedding.forward(token_ids)         # (seq, d_model)

        # Stack of transformer blocks
        for block in self.blocks:
            x = block.forward(x)                      # (seq, d_model)

        # Final layer-norm + language model head
        self._x_after_blocks = x
        x_norm  = self.ln_final.forward(x)            # (seq, d_model)
        logits  = self.lm_head.forward(x_norm)        # (seq, vocab_size)

        return logits

    def backward(self, dlogits: list) -> None:
        """
        Full backward pass through the entire model.

        Args:
            dlogits : gradient of loss w.r.t. logits,  (seq_len, vocab_size)
        """
        # ── LM head ───────────────────────────────────────────────────────────
        dx = self.lm_head.backward(dlogits)            # (seq, d_model)

        # ── Final LayerNorm ───────────────────────────────────────────────────
        dx = self.ln_final.backward(dx)               # (seq, d_model)

        # ── Transformer blocks (reverse order!) ───────────────────────────────
        for block in reversed(self.blocks):
            dx = block.backward(dx)                   # (seq, d_model)

        # ── Embedding layer ───────────────────────────────────────────────────
        self.embedding.backward(dx)                   # scatters into table

    def zero_grad(self) -> None:
        """Reset all gradient accumulators across the model."""
        self.embedding.zero_grad()
        for block in self.blocks:
            block.zero_grad()
        self.ln_final.zero_grad()
        self.lm_head.zero_grad()

    def all_params_and_grads(self) -> list:
        """
        Flatten all (param, grad, name) tuples from every layer.
        The optimiser iterates over this list.
        """
        pg = []
        pg.extend(self.embedding.params_and_grads())
        for i, block in enumerate(self.blocks):
            for p, g, name in block.params_and_grads():
                pg.append((p, g, f"block{i}.{name}"))
        pg.extend(self.ln_final.params_and_grads())
        pg.extend(self.lm_head.params_and_grads())
        return pg

    def count_parameters(self) -> int:
        """Return the total number of scalar parameters in the model."""
        total = 0
        for p, g, name in self.all_params_and_grads():
            if isinstance(p[0], list):
                total += len(p) * len(p[0])
            else:
                total += len(p)
        return total