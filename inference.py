# =============================================================================
#  inference.py  —  Text Generation (Inference)
#
#  Three decoding strategies, all pure Python:
#
#  1. greedy()       — always picks the single highest-probability token.
#                      Fast, deterministic, but repetitive.
#
#  2. sample()       — draws from the full distribution (with temperature).
#                      temperature < 1  →  more focused/conservative
#                      temperature > 1  →  more creative/random
#                      temperature = 1  →  unmodified distribution
#
#  3. top_k_sample() — same as sample() but only considers the top-K tokens.
#                      Prevents extremely unlikely tokens from being chosen.
#
#  All generators share the same autoregressive loop:
#      prompt  →  [encode]  →  feed to model  →  pick next token  →  append  →  repeat
# =============================================================================

import math
import random


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _softmax_1d(x: list) -> list:
    """Numerically stable softmax for a 1-D list."""
    mx = max(x)
    e  = [math.exp(v - mx) for v in x]
    s  = sum(e)
    return [v / s for v in e]

def _apply_temperature(logits: list, temperature: float) -> list:
    """Divide every logit by temperature before softmax."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return [v / temperature for v in logits]

def _greedy_pick(probs: list) -> int:
    """Return the index of the highest probability."""
    best_idx = 0
    best_val = probs[0]
    for i, p in enumerate(probs):
        if p > best_val:
            best_val = p
            best_idx = i
    return best_idx

def _multinomial_sample(probs: list) -> int:
    """
    Draw one sample from a discrete probability distribution.
    Uses the CDF inversion method (no NumPy needed).
    """
    r = random.random()
    cdf = 0.0
    for i, p in enumerate(probs):
        cdf += p
        if r <= cdf:
            return i
    return len(probs) - 1   # fallback for floating-point edge cases

def _top_k_filter(logits: list, k: int) -> list:
    """
    Set all logits outside the top-K to -infinity.
    Returns a new list — original is not modified.
    """
    if k <= 0 or k >= len(logits):
        return logits[:]
    # Find the k-th largest value
    sorted_logits = sorted(logits, reverse=True)
    threshold = sorted_logits[k - 1]
    return [v if v >= threshold else -1e9 for v in logits]

def _last_logits(model, token_ids: list, max_seq_len: int) -> list:
    """
    Run the model on token_ids (truncated to max_seq_len if needed)
    and return the logit vector for the LAST position.
    This is the distribution over 'what token comes next'.
    """
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[-max_seq_len:]   # keep only the most recent context
    logits = model.forward(token_ids)          # (seq_len, vocab_size)
    return logits[-1]                          # last position → next-token logits


# ─────────────────────────────────────────────────────────────────────────────
# Public generation functions
# ─────────────────────────────────────────────────────────────────────────────

def greedy(
    model,
    tokenizer,
    prompt:       str,
    max_new:      int = 100,
    max_seq_len:  int = 128,
    stop_on_eos:  bool = True,
) -> str:
    """
    Greedy decoding: always pick the most probable next token.

    Args:
        model       : trained GPT model
        tokenizer   : CharTokenizer with vocab built
        prompt      : seed text string
        max_new     : maximum number of new tokens to generate
        max_seq_len : context window (truncate if longer)
        stop_on_eos : stop early when EOS token is generated

    Returns:
        Generated string (does NOT include the prompt).
    """
    ids = tokenizer.encode(prompt)

    generated = []
    for _ in range(max_new):
        last = _last_logits(model, ids, max_seq_len)    # (vocab_size,)
        probs = _softmax_1d(last)
        next_id = _greedy_pick(probs)

        if stop_on_eos and next_id == tokenizer.EOS_ID:
            break

        generated.append(next_id)
        ids = ids + [next_id]

    return tokenizer.decode(generated)


def sample(
    model,
    tokenizer,
    prompt:       str,
    max_new:      int   = 100,
    temperature:  float = 0.8,
    max_seq_len:  int   = 128,
    stop_on_eos:  bool  = True,
) -> str:
    """
    Multinomial sampling with temperature control.

    temperature = 0.7  → sharper, more confident output
    temperature = 1.0  → vanilla sampling from model distribution
    temperature = 1.5  → more random and creative

    Args:
        model, tokenizer, prompt, max_new, max_seq_len, stop_on_eos:
            same as greedy()
        temperature : float — controls sharpness of distribution

    Returns:
        Generated string (does NOT include the prompt).
    """
    ids = tokenizer.encode(prompt)

    generated = []
    for _ in range(max_new):
        last    = _last_logits(model, ids, max_seq_len)
        scaled  = _apply_temperature(last, temperature)
        probs   = _softmax_1d(scaled)
        next_id = _multinomial_sample(probs)

        if stop_on_eos and next_id == tokenizer.EOS_ID:
            break

        generated.append(next_id)
        ids = ids + [next_id]

    return tokenizer.decode(generated)


def top_k_sample(
    model,
    tokenizer,
    prompt:       str,
    max_new:      int   = 100,
    k:            int   = 10,
    temperature:  float = 0.8,
    max_seq_len:  int   = 128,
    stop_on_eos:  bool  = True,
) -> str:
    """
    Top-K sampling: restrict the candidate pool to the K most likely tokens,
    then sample with temperature.

    This prevents highly unlikely tokens from ever being chosen, resulting
    in more coherent text while still maintaining variety.

    Args:
        k           : number of top tokens to keep (e.g. 5, 10, 20)
        temperature : sharpness of distribution over the top-K tokens
        others      : same as sample()

    Returns:
        Generated string (does NOT include the prompt).
    """
    ids = tokenizer.encode(prompt)

    generated = []
    for _ in range(max_new):
        last      = _last_logits(model, ids, max_seq_len)
        filtered  = _top_k_filter(last, k)
        scaled    = _apply_temperature(filtered, temperature)
        probs     = _softmax_1d(scaled)
        next_id   = _multinomial_sample(probs)

        if stop_on_eos and next_id == tokenizer.EOS_ID:
            break

        generated.append(next_id)
        ids = ids + [next_id]

    return tokenizer.decode(generated)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    model,
    tokenizer,
    prompt:      str,
    max_new:     int   = 100,
    strategy:    str   = "top_k",    # "greedy" | "sample" | "top_k"
    temperature: float = 0.8,
    k:           int   = 10,
    max_seq_len: int   = 128,
) -> str:
    """
    Unified generation entry point.

    Args:
        strategy : one of "greedy", "sample", "top_k"
        Others   : same as individual functions above.

    Returns:
        Generated text string (prompt + new tokens).
    """
    strategy = strategy.lower()
    if strategy == "greedy":
        new_text = greedy(model, tokenizer, prompt, max_new, max_seq_len)
    elif strategy == "sample":
        new_text = sample(model, tokenizer, prompt, max_new, temperature, max_seq_len)
    elif strategy in ("top_k", "topk"):
        new_text = top_k_sample(model, tokenizer, prompt, max_new, k, temperature, max_seq_len)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         f"Choose from: greedy, sample, top_k")

    return prompt + new_text