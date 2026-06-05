# =============================================================================
#  embeddings.py  —  Token Embeddings + Positional Encoding
#
#  Two distinct pieces glued together:
#
#  1. TokenEmbedding
#     A learnable lookup table:  token_id  →  d_model-dim vector.
#     Backward pass: scatter-adds gradients back to the rows that were read.
#
#  2. PositionalEncoding
#     Fixed sinusoidal pattern injected at every position so the transformer
#     knows WHERE in the sequence each token sits.
#     Formula (Vaswani et al., 2017):
#         PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
#         PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
#     No learnable parameters → no backward pass needed here.
#
#  3. Embedding (combined)
#     x = TokenEmbedding(ids) + PositionalEncoding[:seq_len]
# =============================================================================

import math
from matrix import zeros, randn, zeros_1d, copy_2d, add_2d


# ─────────────────────────────────────────────────────────────────────────────
# 1. TOKEN EMBEDDING TABLE
# ─────────────────────────────────────────────────────────────────────────────

class TokenEmbedding:
    """
    Learnable embedding look-up table.

    Parameters
    ----------
    vocab_size : int   — number of unique tokens
    d_model    : int   — dimension of each embedding vector

    Forward
    -------
    Given token_ids (a Python list of ints), returns a
    (seq_len, d_model) matrix where row i = table[token_ids[i]].

    Backward
    --------
    Receives dL of shape (seq_len, d_model) and scatter-adds each row
    into the corresponding row of d_table.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        self.vocab_size = vocab_size
        self.d_model    = d_model

        # ── Learnable weight matrix ──────────────────────────────────────────
        # Small random init (std = 0.02 matches GPT-2 practice).
        std = 0.02
        self.table  = randn(vocab_size, d_model, std=std)

        # ── Gradient accumulator ─────────────────────────────────────────────
        self.d_table = zeros(vocab_size, d_model)

        # ── Cache filled during forward ──────────────────────────────────────
        self._token_ids: list = []

    def forward(self, token_ids: list) -> list:
        """
        Look up embeddings for each token id.

        Args:
            token_ids : list of int,  shape (seq_len,)

        Returns:
            E : 2-D matrix of shape (seq_len, d_model)
        """
        self._token_ids = token_ids
        return [self.table[i][:] for i in token_ids]   # copy each row

    def backward(self, dL: list) -> None:
        """
        Scatter-add upstream gradient into the embedding table.

        Because the same token can appear multiple times in a sequence,
        we accumulate (+=) rather than assign (=).

        Args:
            dL : upstream gradient, shape (seq_len, d_model)
        """
        for t, tok_id in enumerate(self._token_ids):
            for j in range(self.d_model):
                self.d_table[tok_id][j] += dL[t][j]

    def zero_grad(self) -> None:
        self.d_table = zeros(self.vocab_size, self.d_model)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SINUSOIDAL POSITIONAL ENCODING
class PositionalEncoding:
    def __init__(self, max_seq_len: int, d_model: int) -> None:
        self.d_model = d_model
        self.pe = self._build_table(max_seq_len, d_model)

    @staticmethod
    def _build_table(max_seq_len: int, d_model: int) -> list:
        table = zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(d_model // 2):
                angle = pos / (10000 ** (2 * i / d_model))
                table[pos][2 * i]     = math.sin(angle)
                table[pos][2 * i + 1] = math.cos(angle)
            if d_model % 2 == 1:
                angle = pos / (10000 ** (d_model - 1) / d_model)
                table[pos][d_model - 1] = math.sin(angle)
        return table

    def forward(self, seq_len: int) -> list:
        return [self.pe[pos][:] for pos in range(min(seq_len, len(self.pe)))]
# ─────────────────────────────────────────────────────────────────────────────
# 3. COMBINED EMBEDDING LAYER
# ─────────────────────────────────────────────────────────────────────────────

class Embedding:
    """
    Combines token embeddings and positional encoding into a single layer.

    Forward:
        x = TokenEmbedding(token_ids)  +  PE[:seq_len]   →  (seq_len, d_model)

    Backward:
        Passes dL directly to TokenEmbedding.backward() — the PE has no
        learnable parameters so its gradient contribution is zero and we
        don't need to store it.
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int) -> None:
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(max_seq_len, d_model)

    def forward(self, token_ids: list) -> list:  
        # अगर शब्द 64 से ज्यादा हो जाएं, तो पुराने शब्द हटा दें ताकि क्रैश न हो
        max_len = len(self.pos_enc.pe)
        if len(token_ids) > max_len:
            token_ids = token_ids[-max_len:]
            
        E  = self.token_emb.forward(token_ids)
        PE = self.pos_enc.forward(len(token_ids))
        return add_2d(E, PE)

    def backward(self, dL: list) -> None:
        """
        Backpropagate through the embedding layer.
        dL flows unchanged into the token embedding table.
        The PE contribution has no parameters, so we ignore that branch.
        """
        self.token_emb.backward(dL)

    def zero_grad(self) -> None:
        self.token_emb.zero_grad()

    # ── Expose parameters + gradients for the optimizer ──────────────────────
    def params_and_grads(self) -> list:
        """
        Returns list of (param_matrix, grad_matrix, name) tuples.
        The optimiser iterates over this list to apply updates.
        """
        return [(self.token_emb.table, self.token_emb.d_table, "embed.table")]