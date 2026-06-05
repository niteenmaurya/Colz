import json
import torch
import time  # Generation time track karne ke liye
from flask import Flask, request, jsonify, send_from_directory
from transformer import GPT
from tokenizer import CharTokenizer
from inference import generate 

app = Flask(__name__, static_folder='web')

# 🔥 Safety Hook: CORS bypass (Localhost conflicts se bachne ke liye)
@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
    return response

# 1. Tokenizer Setup (Clean Built-in Load Method)
print("\n⏳ [1/4] Loading tokenizer using vocab.json...")
tok = CharTokenizer()
try:
    tok.load('vocab.json')  # Isse tokenizer built mark ho jayega aur error nahi aayega
    vocab_size = tok.vocab_size
    print(f"✅ Tokenizer loaded successfully. Total Vocab Size: {vocab_size}")
except Exception as e:
    print(f"❌ Error loading vocab.json: {e}")
    vocab_size = 37  # Fallback size

# 2. Load Checkpoint
print("⏳ [2/4] Loading model checkpoint (brain.pt)...")
checkpoint = torch.load('brain.pt', map_location=torch.device('cpu'))

# 3. Get Config
config = checkpoint.get('config', {
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 4,
    'd_ff': 512,
    'max_seq_len': 128
})

# 4. Initialize Model
print("⏳ [3/4] Initializing GPT Architecture...")
model = GPT(
    vocab_size=vocab_size, 
    d_model=config.get('d_model', 128), 
    n_heads=config.get('n_heads', 4), 
    n_layers=config.get('n_layers', 4), 
    d_ff=config.get('d_ff', 512), 
    max_seq_len=config.get('max_seq_len', 128)
)

# 5. Load Weights
print("⏳ [4/4] Injecting trained weights into the model...")
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
print("🚀 [READY] GPT Model is loaded perfectly on CPU! Server starting now...\n")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('web', 'index.html')

@app.route('/generate', methods=['POST', 'OPTIONS'])
def generate_text():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
        
    data = request.json or {}
    prompt = data.get('prompt', '')
    strategy = data.get('strategy', 'top_k')
    k = int(data.get('k', 5))
    temperature = float(data.get('temperature', 0.8))
    max_new = int(data.get('max_new', 50))
    
    # 📝 LIVE TERMINAL LOGGING
    print("="*60)
    print(f"📥 [NEW REQUEST] Frontend se data aaya!")
    print(f"💬 User Prompt : '{prompt}'")
    print(f"⚙️ Settings    : Strategy={strategy} | Max Tokens={max_new} | Temp={temperature} | K={k}")
    print(f"⏳ [THINKING]  AI text generate kar raha hai character-by-character... Please wait...")
    
    start_time = time.time()  # Start timer
    
    try:
        # Asli text generation
        raw_output = generate(model, tok, prompt, max_new=max_new, strategy=strategy, k=k, temperature=temperature)
        
        # ✂️ PROMPT CUTTING LOGIC: Agar output ke aage prompt repeat ho raha hai toh usey kaat do
        final_output = raw_output
        if raw_output.startswith(prompt):
            final_output = raw_output[len(prompt):].strip()
            
        elapsed_time = time.time() - start_time  # Stop timer
        
        print(f"✨ [SUCCESS]   AI ne reply ready kar diya in {elapsed_time:.2f} seconds!")
        print(f"🤖 Raw Output  : {raw_output}")
        print(f"🧼 Clean Reply : {final_output}")
        print("="*60 + "\n")
        
        return jsonify({'result': final_output})

    except Exception as e:
        print(f"❌ [ERROR]      Something went wrong inside the generation loop!")
        print(f"⚠️ Details      : {str(e)}")
        print("="*60 + "\n")
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)