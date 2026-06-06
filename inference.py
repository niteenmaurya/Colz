# =============================================================================
#  inference.py  —  Text Generation
#
#  Strategies:
#    greedy     — highest-prob token, deterministic
#    sample     — multinomial sampling with temperature
#    top_k      — top-K filtered sampling with temperature
#
#  All generation stays on GPU (no Python loop over logits).
#  model.forward() is called with a torch.Tensor directly so the
#  GPU never has to round-trip through Python lists during inference.
# =============================================================================

import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Device (matches transformer.py / trainer.py)
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cpu")
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers  (all tensor-native, no Python loops over vocab)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _get_next_logits(model, ids_tensor: torch.Tensor) -> torch.Tensor:
    """
    Run model on ids_tensor (1, T) and return logits for the LAST position.
    Shape returned: (vocab_size,)
    Stays entirely on DEVICE — no .tolist() round-trip.
    """
    T = ids_tensor.shape[1]
    if T > model.max_seq_len:
        ids_tensor = ids_tensor[:, -model.max_seq_len:]

    # Call model's internal forward directly to avoid the list→tensor overhead
    # in the public forward() compatibility wrapper
    pos    = torch.arange(ids_tensor.shape[1], device=DEVICE).unsqueeze(0)
    x      = model.token_emb(ids_tensor) + model.pos_emb(pos)
    for block in model.blocks:
        x = block(x)
    x      = model.ln_final(x)
    logits = model.lm_head(x)          # (1, T, vocab_size)
    return logits[0, -1, :]            # (vocab_size,)


def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return logits / temperature


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    threshold = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < threshold, float('-inf'))


# ─────────────────────────────────────────────────────────────────────────────
# Core generation loop  (shared by all three strategies)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate_loop(
    model,
    tokenizer,
    prompt:      str,
    max_new:     int,
    temperature: float,
    top_k:       int,         # 0 = disabled
    greedy:      bool,
) -> str:
    model.eval()

    prompt_ids = tokenizer.encode(prompt)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)  # (1, T)

    generated = []

    for _ in range(max_new):
        logits = _get_next_logits(model, ids)          # (vocab_size,) on DEVICE

        if greedy:
            next_id = logits.argmax().item()
        else:
            if top_k > 0:
                logits = _top_k_filter(logits, top_k)
            logits  = _apply_temperature(logits, temperature)
            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

        if next_id == tokenizer.EOS_ID:
            break

        generated.append(next_id)
        ids = torch.cat([
            ids,
            torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)
        ], dim=1)

    return prompt + tokenizer.decode(generated)


# ─────────────────────────────────────────────────────────────────────────────
# Public API  (same signatures as before — drop-in replacement)
# ─────────────────────────────────────────────────────────────────────────────

def greedy(
    model,
    tokenizer,
    prompt:      str,
    max_new:     int = 100,
    max_seq_len: int = 128,
    stop_on_eos: bool = True,
) -> str:
    return _generate_loop(
        model, tokenizer, prompt,
        max_new=max_new, temperature=1.0, top_k=0, greedy=True,
    )


def sample(
    model,
    tokenizer,
    prompt:      str,
    max_new:     int   = 100,
    temperature: float = 0.8,
    max_seq_len: int   = 128,
    stop_on_eos: bool  = True,
) -> str:
    return _generate_loop(
        model, tokenizer, prompt,
        max_new=max_new, temperature=temperature, top_k=0, greedy=False,
    )


def top_k_sample(
    model,
    tokenizer,
    prompt:      str,
    max_new:     int   = 100,
    k:           int   = 10,
    temperature: float = 0.8,
    max_seq_len: int   = 128,
    stop_on_eos: bool  = True,
) -> str:
    return _generate_loop(
        model, tokenizer, prompt,
        max_new=max_new, temperature=temperature, top_k=k, greedy=False,
    )


def generate(
    model,
    tokenizer,
    prompt:      str,
    max_new:     int   = 100,
    strategy:    str   = "top_k",
    temperature: float = 0.8,
    k:           int   = 10,
    max_seq_len: int   = 128,
) -> str:
    """
    Unified entry point. strategy: 'greedy' | 'sample' | 'top_k'
    Returns prompt + generated text.
    """
    strategy = strategy.lower()
    if strategy == "greedy":
        return greedy(model, tokenizer, prompt, max_new)
    elif strategy == "sample":
        return sample(model, tokenizer, prompt, max_new, temperature)
    elif strategy in ("top_k", "topk"):
        return top_k_sample(model, tokenizer, prompt, max_new, k, temperature)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose: greedy | sample | top_k"
        )