# =============================================================================
#  trainer.py  —  Training Engine  (PyTorch GPU version)
#
#  Pure-Python trainer ke saath EXACT same API:
#    trainer = Trainer(model, lr=3e-4)
#    trainer.train(all_ids, seq_len=64, epochs=3000, log_every=100)
#
#  Andar se sab PyTorch — GPU accelerated, autograd, fast.
#  cross_entropy_loss() aur AdamOptimizer classes bhi hain (backward compat).
# =============================================================================

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────────────────
# Device (same as transformer.py)
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = (
    torch.device("cuda")  if torch.cuda.is_available() else
    torch.device("mps")   if torch.backends.mps.is_available() else
    torch.device("cpu")
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CROSS-ENTROPY LOSS  (compatibility wrapper — actual training uses PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def cross_entropy_loss(logits: list, targets: list) -> tuple:
    """
    Kept for backward compatibility.
    In Trainer.train_step() hum directly PyTorch loss use karte hain.
    Agar koi manually call kare toh bhi kaam karega.
    """
    seq_len  = len(logits)
    vocab_sz = len(logits[0])
    PAD_ID   = 0

    logits_t  = torch.tensor(logits,  dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.long)

    # Mask padding
    mask = (targets_t != PAD_ID)
    if mask.sum() == 0:
        zeros = [[0.0] * vocab_sz for _ in range(seq_len)]
        return 0.0, zeros

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    loss    = loss_fn(logits_t, targets_t)

    # Gradient w.r.t. logits (for pure-Python backward compat)
    # PyTorch trainer mein yeh use nahi hota
    import torch.nn.functional as F
    probs   = F.softmax(logits_t, dim=-1).detach()
    n_valid = float(mask.sum())
    dlogits = probs.clone()
    for t in range(seq_len):
        if targets[t] != PAD_ID:
            dlogits[t][targets[t]] -= 1.0
    dlogits /= n_valid
    dlogits[~mask] = 0.0

    return float(loss.item()), dlogits.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 2. ADAM OPTIMISER  (stub — real training uses torch.optim.AdamW)
# ─────────────────────────────────────────────────────────────────────────────

class AdamOptimizer:
    """
    Stub class — API compatible with pure-Python version.
    Real optimizer is torch.optim.AdamW inside Trainer.
    """

    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.wd    = wd
        self.t     = 0

    def step(self, params_and_grads: list) -> None:
        """No-op — Trainer uses torch optimizer directly."""
        self.t += 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRADIENT CLIPPING
# ─────────────────────────────────────────────────────────────────────────────

def clip_gradients(params_and_grads: list, max_norm: float = 1.0) -> float:
    """
    Wrapper around torch.nn.utils.clip_grad_norm_.
    Accepts pure-Python params_and_grads list (ignored — we clip via PyTorch).
    Returns global norm (approx).
    """
    # Actual clipping happens in Trainer.train_step via torch utility
    return max_norm


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    High-level training manager — GPU accelerated via PyTorch.

    Same API as pure-Python version:
        trainer = Trainer(model, lr=3e-4)
        trainer.train(corpus_ids, seq_len=32, epochs=50)
    """

    def __init__(
        self,
        model,
        lr:       float = 3e-4,
        max_norm: float = 1.0,
    ) -> None:
        self.model    = model
        self.max_norm = max_norm

        # Real PyTorch AdamW optimizer
        self.torch_optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        # Compatibility stub
        self.optimizer = AdamOptimizer(lr=lr)

        self.loss_history: list = []
        self._loss_fn = nn.CrossEntropyLoss(ignore_index=0)   # PAD_ID = 0

    # ──────────────────────────────────────────────────────────────────────────
    # train_step — PyTorch autograd version
    # ──────────────────────────────────────────────────────────────────────────

    def train_step(self, token_ids: list) -> float:
        """
        One forward + backward + optimize pass.
        Uses PyTorch autograd — no manual dlogits needed.
        """
        if len(token_ids) < 2:
            return 0.0

        inputs  = token_ids[:-1]
        targets = token_ids[1:]

        # Convert to tensors on GPU
        inp_t = torch.tensor(inputs,  dtype=torch.long,  device=DEVICE)
        tgt_t = torch.tensor(targets, dtype=torch.long,  device=DEVICE)

        # ── Forward (PyTorch graph) ───────────────────────────────────────────
        # We bypass model.forward(list) and call the nn.Module directly
        # to keep the computation graph intact for autograd.
        with torch.enable_grad():
            # model.forward() returns list — we need tensor
            # So we call the underlying PyTorch module directly:
            ids     = inp_t
            seq_len = ids.shape[0]
            if seq_len > self.model.max_seq_len:
                ids = ids[-self.model.max_seq_len:]
                seq_len = self.model.max_seq_len

            pos    = torch.arange(seq_len, device=DEVICE)
            x      = self.model.token_emb(ids) + self.model.pos_emb(pos)
            for block in self.model.blocks:
                x = block(x)
            x      = self.model.ln_final(x)
            logits = self.model.lm_head(x)        # (seq, vocab_size) — tensor

            # ── Loss ─────────────────────────────────────────────────────────
            # logits: (seq, vocab) → CrossEntropyLoss expects (N, C)
            loss = self._loss_fn(logits, tgt_t[:seq_len])

        # ── Backward ─────────────────────────────────────────────────────────
        self.torch_optimizer.zero_grad()
        loss.backward()

        # ── Gradient clipping ─────────────────────────────────────────────────
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

        # ── Optimizer step ────────────────────────────────────────────────────
        self.torch_optimizer.step()

        return float(loss.item())

    # ──────────────────────────────────────────────────────────────────────────
    # train — full training loop
    # ──────────────────────────────────────────────────────────────────────────

    def train(
        self,
        all_token_ids: list,
        seq_len:       int,
        epochs:        int,
        log_every:     int = 10,
    ) -> None:
        """
        Same signature as pure-Python Trainer.train().
        """
        # Build windows
        windows = []
        for start in range(0, len(all_token_ids) - seq_len, seq_len):
            windows.append(all_token_ids[start : start + seq_len + 1])

        if not windows:
            print("[Trainer] Corpus too short for given seq_len.")
            return

        # Learning rate scheduler (cosine decay) — optional but improves results
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.torch_optimizer,
            T_max=epochs,
            eta_min=1e-5,
        )

        total_tokens = len(all_token_ids)
        print(f"[Trainer] corpus={total_tokens} tokens | "
              f"windows={len(windows)} | seq_len={seq_len} | epochs={epochs}")
        print(f"[Trainer] device={DEVICE} | "
              f"model params = {self.model.count_parameters():,}")
        print("-" * 60)

        self.model.train()   # PyTorch train mode (enables dropout if any)

        for epoch in range(1, epochs + 1):
            random.shuffle(windows)
            epoch_loss = 0.0

            for window in windows:
                loss = self.train_step(window)
                epoch_loss += loss

            avg_loss = epoch_loss / len(windows)
            self.loss_history.append(avg_loss)
            scheduler.step()

            if epoch % log_every == 0 or epoch == 1:
                ppl = math.exp(min(avg_loss, 20))
                lr  = self.torch_optimizer.param_groups[0]['lr']
                print(f"  epoch {epoch:4d}/{epochs}  |  "
                      f"loss = {avg_loss:.4f}  |  "
                      f"perplexity = {ppl:.2f}  |  "
                      f"lr = {lr:.2e}")

        print("-" * 60)
        print("[Trainer] Training complete.")
        self.model.eval()   # back to eval mode