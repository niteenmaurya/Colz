# =============================================================================
#  transformer.py  —  GPT-Style Decoder-Only Transformer  (PyTorch version)
#
#  EXACT same API as pure-Python version:
#    model = GPT(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len)
#    logits = model.forward(token_ids)          # token_ids = list of int
#    model.backward(dlogits)                    # NOT used — PyTorch autograd handles this
#    model.zero_grad()
#    model.all_params_and_grads()               # for Adam optimizer compat
#    model.count_parameters()
#
#  main.py mein koi change nahi — bas is file ko transformer.py se replace karo.
# =============================================================================

import torch
import torch.nn as nn
import math


# Device auto-select: CUDA > MPS (Apple) > CPU
DEVICE = (
    torch.device("cuda")  if torch.cuda.is_available() else
    torch.device("mps")   if torch.backends.mps.is_available() else
    torch.device("cpu")
)
print(f"[Transformer] Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# GPT Model
# ─────────────────────────────────────────────────────────────────────────────

class GPT(nn.Module):
    """
    Mini GPT-style decoder-only transformer.
    
    API is compatible with the pure-Python version so main.py works unchanged.
    Internally uses PyTorch for GPU acceleration + autograd.
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
        super().__init__()

        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        # ── Layers ────────────────────────────────────────────────────────────
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (standard GPT trick — reduces params)
        self.lm_head.weight = self.token_emb.weight

        # Init weights like GPT-2
        self.apply(self._init_weights)
        self.to(DEVICE)

        # Internal state for trainer.py compatibility
        self._last_logits_tensor = None

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ──────────────────────────────────────────────────────────────────────────
    # forward() — accepts list of ints (same API as pure Python version)
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, token_ids) -> list:
        """
        Args:
            token_ids : list of int  OR  torch.Tensor (seq_len,)
        
        Returns:
            logits : list of lists  (seq_len, vocab_size)
                     — matches pure-Python API so trainer.py works unchanged
        """
        if isinstance(token_ids, list):
            ids = torch.tensor(token_ids, dtype=torch.long, device=DEVICE)
        else:
            ids = token_ids.to(DEVICE)

        seq_len = ids.shape[0]

        # Clamp to max_seq_len
        if seq_len > self.max_seq_len:
            ids     = ids[-self.max_seq_len:]
            seq_len = self.max_seq_len

        pos = torch.arange(seq_len, device=DEVICE)

        # Embeddings
        x = self.token_emb(ids) + self.pos_emb(pos)   # (seq, d_model)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm + LM head
        x      = self.ln_final(x)
        logits = self.lm_head(x)                       # (seq, vocab_size)

        # Save tensor for trainer (backward needs it)
        self._last_logits_tensor = logits

        # Return as list-of-lists for compatibility with cross_entropy_loss()
        return logits.tolist()

    # ──────────────────────────────────────────────────────────────────────────
    # backward() — called by Trainer.train_step()
    # trainer.py calls: loss, dlogits = cross_entropy_loss(logits, targets)
    #                   model.backward(dlogits)
    # We SKIP the Python cross_entropy and use PyTorch's own loss instead.
    # See trainer.py — we override train_step() there.
    # ──────────────────────────────────────────────────────────────────────────

    def backward(self, dlogits) -> None:
        """
        Not used when trainer.py uses PyTorch loss.
        Kept for API compatibility — does nothing.
        """
        pass

    def zero_grad(self) -> None:
        """Zero all parameter gradients."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def all_params_and_grads(self) -> list:
        """
        Returns list of (param_list, grad_list, name) — matches pure-Python API.
        trainer.py's AdamOptimizer is BYPASSED; PyTorch Adam is used instead.
        This method is kept so count_parameters() and gradient clipping work.
        """
        result = []
        for name, param in self.named_parameters():
            grad = param.grad
            if grad is None:
                grad_list = [[0.0] * param.shape[-1]] * (param.shape[0] if param.dim() > 1 else 1)
            else:
                grad_list = grad.tolist()
            result.append((param.tolist(), grad_list, name))
        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class _TransformerBlock(nn.Module):
    """Pre-LayerNorm decoder block: x = x + Attn(LN(x)); x = x + FFN(LN(x))"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = _FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Causal Self-Attention
# ─────────────────────────────────────────────────────────────────────────────

class _CausalSelfAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, d_model = x.shape

        # QKV projections in one shot
        qkv = self.qkv_proj(x)                             # (seq, 3*D)
        q, k, v = qkv.split(d_model, dim=-1)               # each: (seq, D)

        # Reshape to (n_heads, seq, d_head)
        q = q.view(seq_len, self.n_heads, self.d_head).transpose(0, 1)
        k = k.view(seq_len, self.n_heads, self.d_head).transpose(0, 1)
        v = v.view(seq_len, self.n_heads, self.d_head).transpose(0, 1)

        # Scaled dot-product attention + causal mask
        scale  = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # (h, seq, seq)

        # Causal mask (upper triangle = -inf)
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device),
            diagonal=1
        )
        scores = scores + mask

        attn = torch.softmax(scores, dim=-1)                    # (h, seq, seq)
        out  = torch.matmul(attn, v)                            # (h, seq, d_head)

        # Merge heads
        out = out.transpose(0, 1).contiguous().view(seq_len, d_model)
        return self.out_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class _FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)