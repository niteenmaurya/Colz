# =============================================================================
#  tokenizer.py  —  Character-Level Tokenizer
#
#  Converts raw text  ↔  integer token IDs.
#
#  Why character-level?
#    — Zero dependencies (no byte-pair encoding tables to load)
#    — Vocabulary is tiny (≈ 70–100 chars for English)
#    — Perfect for learning: you can see every single token
#
#  Special tokens:
#    <PAD>  (0)  — padding to uniform sequence length
#    <UNK>  (1)  — unknown character (shouldn't appear after build_vocab)
#    <SOS>  (2)  — start-of-sequence marker for generation
#    <EOS>  (3)  — end-of-sequence marker
# =============================================================================

import json


class CharTokenizer:
    """
    A stateful character-level tokenizer.

    Usage
    -----
    >>> tok = CharTokenizer()
    >>> tok.build_vocab("hello world")
    >>> ids = tok.encode("hello")
    >>> text = tok.decode(ids)
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

    # Convenient integer aliases
    PAD_ID = 0
    UNK_ID = 1
    SOS_ID = 2
    EOS_ID = 3

    def __init__(self) -> None:
        self.char_to_id: dict = {}
        self.id_to_char: dict = {}
        self._built = False

    # ──────────────────────────────────────────────────────────────────────────
    # Vocabulary construction
    # ──────────────────────────────────────────────────────────────────────────

    def build_vocab(self, corpus: str) -> None:
        """
        Scan `corpus` and assign a unique integer ID to every distinct
        character.  Special tokens always occupy IDs 0–3.

        Args:
            corpus: The full training text (one big string).
        """
        self.char_to_id = {}
        self.id_to_char = {}

        # Reserve slots 0–3 for special tokens
        for idx, tok in enumerate(self.SPECIAL_TOKENS):
            self.char_to_id[tok] = idx
            self.id_to_char[idx] = tok

        # Add every unique character found in the corpus
        current_id = len(self.SPECIAL_TOKENS)   # starts at 4
        for ch in sorted(set(corpus)):           # sorted → deterministic vocab
            if ch not in self.char_to_id:
                self.char_to_id[ch] = current_id
                self.id_to_char[current_id] = ch
                current_id += 1

        self._built = True

    # ──────────────────────────────────────────────────────────────────────────
    # Encoding  (text → ids)
    # ──────────────────────────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_sos: bool = False,
        add_eos: bool = False,
    ) -> list:
        """
        Convert a string into a list of integer token IDs.

        Unknown characters map to UNK_ID so encoding never raises.

        Args:
            text    : The string to encode.
            add_sos : Prepend SOS_ID before the sequence.
            add_eos : Append  EOS_ID after  the sequence.

        Returns:
            List of int token IDs.
        """
        self._require_built()
        ids = [self.char_to_id.get(ch, self.UNK_ID) for ch in text]
        if add_sos:
            ids = [self.SOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        return ids

    def encode_padded(self, text: str, seq_len: int) -> list:
        """
        Encode text and pad (or truncate) to exactly `seq_len` tokens.
        Padding is added on the right with PAD_ID.
        """
        ids = self.encode(text)
        if len(ids) >= seq_len:
            return ids[:seq_len]
        return ids + [self.PAD_ID] * (seq_len - len(ids))

    # ──────────────────────────────────────────────────────────────────────────
    # Decoding  (ids → text)
    # ──────────────────────────────────────────────────────────────────────────

    def decode(self, ids: list, skip_special: bool = True) -> str:
        """
        Convert a list of integer token IDs back into a human-readable string.

        Args:
            ids           : List of int token IDs.
            skip_special  : If True, drop PAD / UNK / SOS / EOS from output.

        Returns:
            Decoded string.
        """
        self._require_built()
        special_ids = {self.PAD_ID, self.UNK_ID, self.SOS_ID, self.EOS_ID}
        chars = []
        for i in ids:
            ch = self.id_to_char.get(i, self.UNK_TOKEN)
            if skip_special and i in special_ids:
                continue
            chars.append(ch)
        return "".join(chars)

    # ──────────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Total number of tokens including special tokens."""
        return len(self.char_to_id)

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError(
                "Tokenizer vocabulary not built.  "
                "Call build_vocab(corpus) first."
            )

    def save(self, path: str) -> None:
        """Persist the vocabulary to a JSON file."""
        data = {
            "char_to_id": self.char_to_id,
            "id_to_char": {str(k): v for k, v in self.id_to_char.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Restore a vocabulary from a JSON file saved with save()."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        self._built = True

    def __repr__(self) -> str:
        status = f"vocab_size={self.vocab_size}" if self._built else "not built"
        return f"CharTokenizer({status})"