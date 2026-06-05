# =============================================================================
#  main.py  —  The Ultimate GPT-Standard Fine-Tuning Architecture
# =============================================================================

import json
import torch
from tokenizer import CharTokenizer
from transformer import GPT
from trainer import Trainer
from inference import generate

# ─── 1. Load and Format Dataset (The Official OpenAI ChatML Way) ──────────────
DATASET_PATH = r"C:\Users\nitee\colz\dataset.jsonl"
corpus_pieces = []

print(f"⏳ GPT Engine: Initializing data pipeline from: {DATASET_PATH}")

# यह वो सीक्रेट सिस्टम प्रॉम्ट है जो बड़े GPT मॉडल्स को कंट्रोल करता है
SYSTEM_PROMPT = "You are a friendly, helpful, and witty AI assistant."

try:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                
                # 1. हर नई बातचीत की शुरुआत में System Instruction इंजेक्ट करें (GPT Standard)
                corpus_pieces.append(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
                
                # 2. उसके बाद यूजर और असिस्टेंट की बातचीत जोड़ें
                for msg in data.get("messages", []):
                    role = msg.get("role", "").strip().lower()  # 'user' या 'assistant'
                    content = msg.get("content", "").strip()
                    
                    corpus_pieces.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping malformed JSON on line {line_num}: {e}")

    CORPUS = "".join(corpus_pieces)
    print(f"✅ GPT Engine: Dataset perfectly structured! Total length: {len(CORPUS):,} chars.")

except FileNotFoundError:
    raise FileNotFoundError(f"❌ Checkpoint error: dataset.jsonl नहीं मिली। पाथ चेक करें।")

# ─── 2. Tokenizer ─────────────────────────────────────────────────────────────
tok = CharTokenizer()
tok.build_vocab(CORPUS)
tok.save("vocab.json")
print(f"Vocab size: {tok.vocab_size}")

all_ids = tok.encode(CORPUS)

# ─── 3. Model Config (Optimized Context Window) ───────────────────────────────
# स्पेशल टोकन्स और सिस्टम प्रॉम्ट की वजह से हमने मेमोरी (Context) को 256 रखा है
MAX_LEN = 256

model = GPT(
    vocab_size  = tok.vocab_size,
    d_model     = 128,    
    n_heads     = 4,
    n_layers    = 4,      
    d_ff        = 512,    
    max_seq_len = MAX_LEN,
)
print(f"Model parameters: {model.count_parameters():,}")

# ─── 4. Train ─────────────────────────────────────────────────────────────────
trainer = Trainer(model, lr=3e-4)
trainer.train(all_ids, seq_len=MAX_LEN, epochs=3000, log_every=100)

# ─── 5. GPT-Level Live Test Generation ────────────────────────────────────────
model.eval()

# प्रॉम्ट देते समय भी हमें पहले System बताना होगा, फिर User... तब जाकर Assistant खुद बोलेगा!
prompts = [
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n",
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nAny plans for the weekend?<|im_end|>\n<|im_start|>assistant\n"
]

print("\n--- Running Official GPT-Level Test Generation ---")
for p in prompts:
    out = generate(model, tok, p, max_new=60, strategy="top_k", k=5, temperature=0.8)
    print(f"\nGenerated Output:\n{out}")
    print("-" * 60)

# ─── 6. Save Brain ────────────────────────────────────────────────────────────
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': tok.char_to_id,
    'config': {
        'vocab_size':  tok.vocab_size,
        'd_model':     128,
        'n_heads':     4,
        'n_layers':    4,
        'd_ff':        512,
        'max_seq_len': MAX_LEN,
    }
}, "brain.pt")
print("\n🔥 Big-Level GPT Brain saved to 'brain.pt'. Now you are ready to launch!")