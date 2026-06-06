# =============================================================================
#  main.py  —  GPT Fine-Tuning Entry Point
# =============================================================================

import json
import torch
import torch._dynamo
from tokenizer  import CharTokenizer
from transformer import GPT
from trainer    import Trainer
from inference  import generate

# ─────────────────────────────────────────────────────────────────────────────
# 0. Single config dict  —  change values HERE only, nowhere else
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "dataset_path" : r"C:\Users\nitee\colz\dataset.jsonl",
    "d_model"      : 256,
    "n_heads"      : 8,
    "n_layers"     : 6,
    "d_ff"         : 1024,
    "max_seq_len"  : 128,
    "lr"           : 3e-4,
    "epochs"       : 3000,
    "log_every"    : 100,
    "batch_size"   : 256,
    "vocab_save"   : "vocab.json",
    "brain_save"   : "brain.pt",
    "use_compile"  : False,   # True on Colab (PyTorch 2.0+), False on Windows
}

SYSTEM_PROMPT = "You are a friendly, helpful, and witty AI assistant."

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load + format dataset
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n⏳ [1/5] Loading dataset: {CONFIG['dataset_path']}")

corpus_pieces = []
try:
    with open(CONFIG["dataset_path"], "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                corpus_pieces.append(
                    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                )
                for msg in data.get("messages", []):
                    role    = msg.get("role", "").strip().lower()
                    content = msg.get("content", "").strip()
                    corpus_pieces.append(
                        f"<|im_start|>{role}\n{content}<|im_end|>\n"
                    )
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Skipping malformed JSON line {line_num}: {e}")

except FileNotFoundError:
    raise FileNotFoundError(
        f"❌ dataset.jsonl not found at: {CONFIG['dataset_path']}"
    )

CORPUS = "".join(corpus_pieces)
print(f"✅ Corpus ready — {len(CORPUS):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
print("\n⏳ [2/5] Building vocabulary...")
tok = CharTokenizer()
tok.build_vocab(CORPUS)
tok.save(CONFIG["vocab_save"])
print(f"✅ Vocab size: {tok.vocab_size}  →  saved to {CONFIG['vocab_save']}")

all_ids = tok.encode(CORPUS)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Model
# ─────────────────────────────────────────────────────────────────────────────
print("\n⏳ [3/5] Building model...")
model = GPT(
    vocab_size  = tok.vocab_size,
    d_model     = CONFIG["d_model"],
    n_heads     = CONFIG["n_heads"],
    n_layers    = CONFIG["n_layers"],
    d_ff        = CONFIG["d_ff"],
    max_seq_len = CONFIG["max_seq_len"],
)
print(f"✅ Params: {model.count_parameters():,}")

# torch.compile — set use_compile=True on Colab, keep False on Windows
if CONFIG["use_compile"]:
    torch._dynamo.config.suppress_errors = True
    model = torch.compile(model)
    print("✅ torch.compile active — extra ~1.5-2x speed")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Train
# ─────────────────────────────────────────────────────────────────────────────
print("\n⏳ [4/5] Starting training...")
trainer = Trainer(
    model,
    lr         = CONFIG["lr"],
    batch_size = CONFIG["batch_size"],
)
trainer.train(
    all_ids,
    seq_len   = CONFIG["max_seq_len"],
    epochs    = CONFIG["epochs"],
    log_every = CONFIG["log_every"],
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Quick generation test
# ─────────────────────────────────────────────────────────────────────────────
print("\n⏳ [5/5] Generation test...")
model.eval()

test_prompts = [
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\nHi!<|im_end|>\n"
    f"<|im_start|>assistant\n",

    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\nAny plans for the weekend?<|im_end|>\n"
    f"<|im_start|>assistant\n",
]

print("\n" + "─" * 60)
for p in test_prompts:
    out = generate(
        model, tok, p,
        max_new=60, strategy="top_k", k=5, temperature=0.8,
    )
    # print only the assistant reply, not the full prompt
    reply = out[len(p):].strip()
    user_line = p.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0].strip()
    print(f"User : {user_line}")
    print(f"Bot  : {reply}")
    print("─" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save
# ─────────────────────────────────────────────────────────────────────────────
torch.save({
    "model_state_dict" : model.state_dict(),
    "vocab"            : tok.char_to_id,
    "config"           : {
        "vocab_size"  : tok.vocab_size,
        "d_model"     : CONFIG["d_model"],
        "n_heads"     : CONFIG["n_heads"],
        "n_layers"    : CONFIG["n_layers"],
        "d_ff"        : CONFIG["d_ff"],
        "max_seq_len" : CONFIG["max_seq_len"],
    },
    "loss_history" : trainer.loss_history,
}, CONFIG["brain_save"])

print(f"\n🔥 Brain saved → {CONFIG['brain_save']}")
print("🚀 Done!")