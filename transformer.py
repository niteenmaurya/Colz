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
    # forward() — accepts list of ints OR torch.Tensor
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, token_ids) -> list:
        """
        Args:
            token_ids : list of int  OR  torch.Tensor (batch, seq_len)
        
        Returns:
            logits : list of lists  (seq_len, vocab_size) or torch.Tensor
        """
        is_list = isinstance(token_ids, list)
        if is_list:
            ids = torch.tensor(token_ids, dtype=torch.long, device=DEVICE)
        else:
            ids = token_ids.to(DEVICE)

        # Safe 2D conversion for single sequences
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        B, T = ids.shape

        # Clamp to max_seq_len
        if T > self.max_seq_len:
            ids = ids[:, -self.max_seq_len:]
            T = self.max_seq_len

        pos = torch.arange(T, device=DEVICE).unsqueeze(0)  # Shape: (1, T)

        # Embeddings (pos matches broad-casting rules)
        x = self.token_emb(ids) + self.pos_emb(pos)   # (B, T, d_model)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm + LM head
        x      = self.ln_final(x)
        logits = self.lm_head(x)                      # (B, T, vocab_size)

        # Save tensor for trainer (backward compatibility)
        self._last_logits_tensor = logits

        # If input was list, return as list-of-lists (matching old compatibility specs)
        if is_list:
            return logits[0].tolist()
        return logits

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
# Causal Self-Attention (Fully Vectorized 3D Batch Compatible)
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
        B, T, C = x.shape  # Supports 3D Batch Shape directly!

        # QKV projections in one shot
        qkv = self.qkv_proj(x)                                 # (B, T, 3*C)
        
        # 🔥 FIX: chunk(3) ensures 3 equal tensors regardless of shape variations
        q, k, v = qkv.chunk(3, dim=-1)                         # each: (B, T, C)

        # Reshape to (B, n_heads, T, d_head) for parallel multi-head attention
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention + causal mask
        scale  = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, h, T, T)

        # Causal mask (upper triangle = -inf)
        mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device),
            diagonal=1
        )
        scores = scores + mask

        attn = torch.softmax(scores, dim=-1)                    # (B, h, T, T)
        out  = torch.matmul(attn, v)                            # (B, h, T, d_head)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
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