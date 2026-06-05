# =============================================================================
#  main.py  —  Training Entry Point
#  Yahan corpus define karo, model train karo, phir generate karo.
# =============================================================================

from tokenizer import CharTokenizer
from transformer import GPT
from trainer import Trainer
from inference import generate

# ─── 1. Corpus (training data) ───────────────────────────────────────────────
with open("data.txt", "r", encoding="utf-8") as f:
    CORPUS = f.read()

# ─── 2. Tokenizer ─────────────────────────────────────────────────────────────
tok = CharTokenizer()
tok.build_vocab(CORPUS)
tok.save("vocab.json")
print(f"Vocab size: {tok.vocab_size}")

all_ids = tok.encode(CORPUS)

# ─── 3. Model config ──────────────────────────────────────────────────────────
model = GPT(
    vocab_size  = tok.vocab_size,
    d_model     = 64,
    n_heads     = 4,
    n_layers    = 2,
    d_ff        = 256,
    max_seq_len = 64,
)
print(f"Model parameters: {model.count_parameters():,}")

# ─── 4. Train ─────────────────────────────────────────────────────────────────
trainer = Trainer(model, lr=3e-4)
trainer.train(all_ids, seq_len=64, epochs=3000, log_every=100)

# ─── 5. Test generation ───────────────────────────────────────────────────────
# ─── 5. Test generation ───────────────────────────────────────────────────────
prompts = ["the cat", "the dog", "a cat"]
for p in prompts:
    out = generate(model, tok, p, max_new=40, strategy="top_k", k=5)
    print(f"\nPrompt: '{p}'\nGenerated: {out}")

import pickle

# --- ट्रेनिंग के बाद दिमाग सेव करें ---
with open("brain.pkl", "wb") as f:
    pickle.dump(model, f)
print("AI का दिमाग brain.pkl में सेव हो गया है!")