# =============================================================================
#  trainer.py  —  Training Engine
#
#  Contains:
#    1. cross_entropy_loss()  — loss computation + dlogits for backward
#    2. AdamOptimizer         — Adam update rule (pure Python)
#    3. Trainer               — full training loop with gradient clipping
#
#  The training objective is next-token prediction:
#    Given tokens [t0, t1, …, t_{n-1}], predict [t1, t2, …, t_n].
#    At each position i the model produces a logit vector over the vocabulary,
#    and we compute cross-entropy against the true next token.
# =============================================================================

import math
import random
from matrix import zeros, zeros_1d, global_l2_norm, scale_2d
# ─────────────────────────────────────────────────────────────────────────────
# 1. CROSS-ENTROPY LOSS
# ─────────────────────────────────────────────────────────────────────────────

def cross_entropy_loss(logits: list, targets: list) -> tuple:
    """
    Compute cross-entropy loss and its gradient w.r.t. logits.

    For each position t:
        probs[t]  = softmax(logits[t])
        loss[t]   = -log(probs[t][targets[t]])   (negative log-likelihood)

    Combined gradient (softmax + cross-entropy, Jacobian simplifies to):
        d(loss)/d(logits[t]) = (probs[t] - one_hot(targets[t])) / seq_len

    Args:
        logits  : (seq_len, vocab_size)  — raw model output
        targets : list of int, shape (seq_len,)  — ground-truth next tokens

    Returns:
        (loss_value: float, dlogits: list)
        where dlogits has same shape as logits.

    Note: Positions where targets[t] == 0 (PAD_ID) are masked to 0.
    """
    seq_len   = len(logits)
    vocab_sz  = len(logits[0])
    PAD_ID    = 0

    total_loss = 0.0
    n_valid    = 0         # count of non-padded positions
    dlogits    = zeros(seq_len, vocab_sz)

    for t in range(seq_len):
        if targets[t] == PAD_ID:
            continue      # skip padding positions

        row  = logits[t]

        # Numerically stable softmax
        mx   = max(row)
        e    = [math.exp(v - mx) for v in row]
        s    = sum(e)
        prob = [v / s for v in e]

        # Cross-entropy for this position
        p_correct   = max(prob[targets[t]], 1e-12)   # clamp to avoid log(0)
        total_loss -= math.log(p_correct)
        n_valid    += 1

        # Gradient: prob - one_hot(target)
        for j in range(vocab_sz):
            dlogits[t][j] = prob[j]
        dlogits[t][targets[t]] -= 1.0

    # Average loss over valid positions
    if n_valid == 0:
        return 0.0, zeros(seq_len, vocab_sz)

    avg_loss = total_loss / n_valid

    # Normalise gradient by n_valid (same normalisation as average loss)
    for t in range(seq_len):
        for j in range(vocab_sz):
            dlogits[t][j] /= n_valid

    return avg_loss, dlogits


# ─────────────────────────────────────────────────────────────────────────────
# 2. ADAM OPTIMISER
# ─────────────────────────────────────────────────────────────────────────────

class AdamOptimizer:
    """
    Adam optimiser (Kingma & Ba, 2015).

    Update rule per parameter θ:
        m_t = β1·m_{t-1} + (1-β1)·g
        v_t = β2·v_{t-1} + (1-β2)·g²
        m̂  = m_t / (1 - β1^t)           ← bias correction
        v̂  = v_t / (1 - β2^t)
        θ  ← θ - lr · m̂ / (√v̂ + ε)

    Parameters are identified by a string key (the `name` field from
    params_and_grads()).  Moment matrices are lazily initialised on first use.

    Supports both 2-D weight matrices and 1-D bias/gamma/beta vectors.
    """

    def __init__(
        self,
        lr:     float = 3e-4,
        beta1:  float = 0.9,
        beta2:  float = 0.999,
        eps:    float = 1e-8,
        wd:     float = 0.01,   # weight-decay (L2 regularisation)
    ) -> None:
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.wd    = wd
        self.t     = 0          # global step counter

        # First and second moment estimates keyed by parameter name
        self._m: dict = {}
        self._v: dict = {}

    def _init_moments(self, name: str, param) -> None:
        """Lazily initialise moment accumulators to zero."""
        if name not in self._m:
            if isinstance(param[0], list):
                rows = len(param);  cols = len(param[0])
                self._m[name] = zeros(rows, cols)
                self._v[name] = zeros(rows, cols)
            else:
                n = len(param)
                self._m[name] = zeros_1d(n)
                self._v[name] = zeros_1d(n)

    def step(self, params_and_grads: list) -> None:
        """
        Apply one Adam update to every parameter in `params_and_grads`.

        Args:
            params_and_grads : list of (param, grad, name) tuples.
                               param and grad are modified IN PLACE.
        """
        self.t += 1
        b1, b2, eps, lr, wd = self.beta1, self.beta2, self.eps, self.lr, self.wd

        # Bias-correction multipliers (computed once per step)
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t

        for param, grad, name in params_and_grads:
            self._init_moments(name, param)
            m = self._m[name]
            v = self._v[name]

            is_2d = isinstance(param[0], list)

            if is_2d:
                rows = len(param);  cols = len(param[0])
                for i in range(rows):
                    for j in range(cols):
                        g = grad[i][j]

                        # Weight decay (not applied to biases / norms)
                        if wd > 0 and "bias" not in name and "beta" not in name:
                            g += wd * param[i][j]

                        # Moment updates
                        m[i][j] = b1 * m[i][j] + (1.0 - b1) * g
                        v[i][j] = b2 * v[i][j] + (1.0 - b2) * g * g

                        # Bias-corrected estimates
                        m_hat = m[i][j] / bc1
                        v_hat = v[i][j] / bc2

                        # Parameter update
                        param[i][j] -= lr * m_hat / (math.sqrt(v_hat) + eps)
            else:
                # 1-D vector (bias, gamma, beta)
                n = len(param)
                for j in range(n):
                    g = grad[j]

                    m[j] = b1 * m[j] + (1.0 - b1) * g
                    v[j] = b2 * v[j] + (1.0 - b2) * g * g

                    m_hat = m[j] / bc1
                    v_hat = v[j] / bc2

                    param[j] -= lr * m_hat / (math.sqrt(v_hat) + eps)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRADIENT CLIPPING
# ─────────────────────────────────────────────────────────────────────────────

def clip_gradients(params_and_grads: list, max_norm: float = 1.0) -> float:
    """
    Global gradient clipping by L2 norm.

    Collects every gradient scalar across all parameters, computes the
    global L2 norm, and if it exceeds `max_norm`, scales all gradients
    down proportionally.

    Returns the global norm (before clipping) — useful for monitoring.
    """
    # Compute global norm
    total_sq = 0.0
    for _, grad, _ in params_and_grads:
        if isinstance(grad[0], list):
            for row in grad:
                total_sq += sum(v * v for v in row)
        else:
            total_sq += sum(v * v for v in grad)

    global_norm = math.sqrt(total_sq)

    if global_norm > max_norm:
        scale = max_norm / (global_norm + 1e-8)
        for _, grad, _ in params_and_grads:
            if isinstance(grad[0], list):
                for row in grad:
                    for j in range(len(row)):
                        row[j] *= scale
            else:
                for j in range(len(grad)):
                    grad[j] *= scale

    return global_norm


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    High-level training manager.

    Usage
    -----
    >>> trainer = Trainer(model, lr=3e-4)
    >>> trainer.train(corpus_ids, seq_len=32, epochs=50)
    """

    def __init__(
        self,
        model,
        lr:       float = 3e-4,
        max_norm: float = 1.0,
    ) -> None:
        self.model     = model
        self.optimizer = AdamOptimizer(lr=lr)
        self.max_norm  = max_norm
        self.loss_history: list = []

    # ──────────────────────────────────────────────────────────────────────────
    # Single training step
    # ──────────────────────────────────────────────────────────────────────────

    def train_step(self, token_ids: list) -> float:
        """
        One forward + backward + optimise pass on a single sequence.

        The input sequence is:   token_ids[:-1]
        The target sequence is:  token_ids[1:]

        Returns the scalar loss for this step.
        """
        inputs  = token_ids[:-1]   # all tokens except last
        targets = token_ids[1:]    # all tokens except first

        if len(inputs) < 1:
            return 0.0

        # ── 1. Forward pass ───────────────────────────────────────────────────
        logits = self.model.forward(inputs)               # (seq-1, vocab_size)

        # ── 2. Loss + gradient of loss w.r.t. logits ─────────────────────────
        loss, dlogits = cross_entropy_loss(logits, targets)

        # ── 3. Zero gradients before backward ────────────────────────────────
        self.model.zero_grad()

        # ── 4. Backward pass ──────────────────────────────────────────────────
        self.model.backward(dlogits)

        # ── 5. Gradient clipping ──────────────────────────────────────────────
        pg = self.model.all_params_and_grads()
        clip_gradients(pg, self.max_norm)

        # ── 6. Optimiser step ─────────────────────────────────────────────────
        self.optimizer.step(pg)

        return loss

    # ──────────────────────────────────────────────────────────────────────────
    # Full training loop
    # ──────────────────────────────────────────────────────────────────────────

    def train(
        self,
        all_token_ids: list,
        seq_len:       int,
        epochs:        int,
        log_every:     int = 10,
    ) -> None:
        """
        Train the model over multiple epochs.

        The corpus is split into non-overlapping windows of length seq_len+1
        (input = first seq_len tokens, target = last seq_len tokens, shifted by 1).

        Args:
            all_token_ids : flat list of int — the full encoded corpus
            seq_len       : context window size
            epochs        : number of full passes over the corpus
            log_every     : print a log line every N epochs
        """
        # Build training windows (seq_len + 1 to include the target token)
        windows = []
        step = seq_len             # non-overlapping stride
        for start in range(0, len(all_token_ids) - seq_len, step):
            windows.append(all_token_ids[start : start + seq_len + 1])

        if not windows:
            print("[Trainer] Corpus too short for the given seq_len.")
            return

        total_tokens = len(all_token_ids)
        print(f"[Trainer] corpus={total_tokens} tokens | "
              f"windows={len(windows)} | seq_len={seq_len} | epochs={epochs}")
        print(f"[Trainer] model params = {self.model.count_parameters():,}")
        print("-" * 60)

        for epoch in range(1, epochs + 1):
            random.shuffle(windows)          # shuffle each epoch
            epoch_loss = 0.0

            for window in windows:
                loss = self.train_step(window)
                epoch_loss += loss

            avg_loss = epoch_loss / len(windows)
            self.loss_history.append(avg_loss)

            if epoch % log_every == 0 or epoch == 1:
                # Perplexity = e^(avg_loss)  — lower is better
                ppl = math.exp(min(avg_loss, 20))   # clamp to avoid overflow
                print(f"  epoch {epoch:4d}/{epochs}  |  "
                      f"loss = {avg_loss:.4f}  |  "
                      f"perplexity = {ppl:.2f}")

        print("-" * 60)
        print("[Trainer] Training complete.")