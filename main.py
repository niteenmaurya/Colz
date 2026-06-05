# =============================================================================
#  main.py  —  Training Entry Point  (PyTorch GPU version)
#  Koi change nahi tha — same API, GPU mein train hoga automatically.
# =============================================================================

from tokenizer import CharTokenizer
from transformer import GPT
from trainer import Trainer
from inference import generate
import pickle
import torch

# ─── 1. Corpus ────────────────────────────────────────────────────────────────
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
    d_model     = 128,    # 64 se badhaaya — better quality
    n_heads     = 4,
    n_layers    = 4,      # 2 se 4 — deeper network
    d_ff        = 512,    # 256 se badhaaya
    max_seq_len = 128,
)
print(f"Model parameters: {model.count_parameters():,}")

# ─── 4. Train ─────────────────────────────────────────────────────────────────
trainer = Trainer(model, lr=3e-4)
trainer.train(all_ids, seq_len=128, epochs=3000, log_every=100)

# ─── 5. Test generation ───────────────────────────────────────────────────────
model.eval()
prompts = ["the cat", "the dog", "a cat", "once upon"]
for p in prompts:
    out = generate(model, tok, p, max_new=60, strategy="top_k", k=5, temperature=0.8)
    print(f"\nPrompt: '{p}'\nGenerated: {out}")

# ─── 6. Save brain ────────────────────────────────────────────────────────────
# PyTorch ka proper tarika — pickle se better
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': tok.char_to_id,
    'config': {
        'vocab_size':  tok.vocab_size,
        'd_model':     128,
        'n_heads':     4,
        'n_layers':    4,
        'd_ff':        512,
        'max_seq_len': 128,
    }
}, "brain.pt")
print("\nAI ka dimaag brain.pt mein save ho gaya!")

# Pickle bhi save kar lo (purana tarika, compatibility ke liye)
with open("brain.pkl", "wb") as f:
    pickle.dump(model, f)
print("Pickle backup: brain.pkl bhi save hua!")