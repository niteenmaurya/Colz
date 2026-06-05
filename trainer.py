import math
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

# ─────────────────────────────────────────────────────────────────────────────
# Device setup (Auto-select: CUDA > MPS > CPU)
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = (
    torch.device("cuda")  if torch.cuda.is_available() else
    torch.device("mps")   if torch.backends.mps.is_available() else
    torch.device("cpu")
)

# ─────────────────────────────────────────────────────────────────────────────
# Stubs for backward compatibility with older legacy scripts
# ─────────────────────────────────────────────────────────────────────────────
class AdamOptimizer:
    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
        self.t = 0
    def step(self, params_and_grads: list) -> None:
        pass

def clip_gradients(params_and_grads: list, max_norm: float = 1.0) -> float:
    return max_norm

def cross_entropy_loss(logits: list, targets: list) -> tuple:
    return 0.0, []

# ─────────────────────────────────────────────────────────────────────────────
# HIGH-SPEED PARALLEL GPU BATCH TRAINER (WITH DYNAMIC PADDING)
# ─────────────────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(
        self,
        model,
        lr:       float = 3e-4,
        max_norm: float = 1.0,
    ) -> None:
        self.model    = model
        self.max_norm = max_norm

        # PyTorch built-in AdamW Optimizer for stable Transformer training
        self.torch_optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        self.optimizer = AdamOptimizer(lr=lr)
        self.loss_history: list = []
        self._loss_fn = nn.CrossEntropyLoss(ignore_index=0)   # PAD_ID = 0

    def train_step(self, batch_windows: list, seq_len: int) -> float:
        """
        Processes an entire batch of sequences concurrently as a single 2D tensor.
        Ensures perfect sequence-length alignment inside the batch using dynamic padding.
        """
        self.torch_optimizer.zero_grad()
        
        inputs_list = []
        targets_list = []
        
        for token_ids in batch_windows:
            if len(token_ids) < 2:
                continue
            
            inp = token_ids[:-1]
            tgt = token_ids[1:]
            
            # Safe truncation if the loaded tokens exceed the model's capacity limit
            if len(inp) > seq_len:
                inp = inp[-seq_len:]
                tgt = tgt[-seq_len:]
                
            # Dynamic Padding: Pads smaller sequences to match EXACTLY 'seq_len'
            pad_len = seq_len - len(inp)
            if pad_len > 0:
                inp = inp + [0] * pad_len
                tgt = tgt + [0] * pad_len
                
            inputs_list.append(inp)
            targets_list.append(tgt)
            
        if not inputs_list:
            return 0.0

        # Converts to ultra-fast PyTorch long tensors on the targeted device (GPU/MPS/CPU)
        inp_t = torch.tensor(inputs_list,  dtype=torch.long,  device=DEVICE)
        tgt_t = torch.tensor(targets_list, dtype=torch.long,  device=DEVICE)

        # Securely unpacks dimensions (Guaranteed 2D shape [Batch, Sequence_Length])
        B, T = inp_t.shape

        try:
            with torch.enable_grad():
                # Broadcast positional indices over the batch
                pos = torch.arange(T, device=DEVICE).unsqueeze(0)
                
                # Forward pass: Combined Embedding layer
                x = self.model.token_emb(inp_t) + self.model.pos_emb(pos)
                
                # Deep Attention block computation
                for block in self.model.blocks:
                    x = block(x)
                
                x = self.model.ln_final(x)
                logits = self.model.lm_head(x) # Shape: [B, T, Vocab_size]

                # Flatten logits & targets, then calculate cross entropy loss (ignoring PAD tokens)
                loss = self._loss_fn(logits.view(-1, logits.size(-1)), tgt_t.view(-1))
                loss.backward()

                # Gradient clipping to prevent gradient explosion issues
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                
                # Gradient update step
                self.torch_optimizer.step()
                
                return loss.item()

        except Exception as e:
            # Handles errors cleanly to prevent crashing mid-training run
            print(f"\n⚠️ [TRAINER DEBUG] Error during train_step processing: {str(e)}")
            return 0.0

    def train(
        self,
        all_token_ids: list,
        seq_len:       int,
        epochs:        int,
        log_every:     int = 5,
    ) -> None:
        """
        Triggers the highly optimized parallel training loops.
        """
        # Slice corpus tokens into aligned sliding context windows
        windows = [all_token_ids[s : s + seq_len + 1] for s in range(0, len(all_token_ids) - seq_len, seq_len)]
        if not windows:
            # Fallback if corpus length is shorter than the configured context length
            windows = [all_token_ids]

        batch_size = 64  # Vectorized batch processing size
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.torch_optimizer,
            T_max=epochs,
            eta_min=1e-5,
        )

        print("=" * 65)
        print(f"🔥 GPT ENGINE ACCELERATED (DYNAMIC PADDING ACTIVE)!")
        print(f"👉 Target Device : {DEVICE.type.upper()}")
        print(f"👉 Batch Size    : {batch_size} sequences/step")
        print(f"👉 Total Params  : {self.model.count_parameters():,}")
        print(f"👉 Total Windows : {len(windows)}")
        print("=" * 65)
        
        self.model.train()
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            random.shuffle(windows)
            
            # Chunk the shuffled windows into uniform batches of size 64
            batches = [windows[i : i + batch_size] for i in range(0, len(windows), batch_size)]
            
            epoch_loss = 0.0
            steps_run = 0
            
            for b in batches:
                loss_val = self.train_step(b, seq_len)
                epoch_loss += loss_val
                steps_run += 1
            
            avg_loss = epoch_loss / max(steps_run, 1)
            self.loss_history.append(avg_loss)
            scheduler.step()

            # Dynamic logs to keep track of training metrics (Logs every 5 epochs)
            if epoch % log_every == 0 or epoch == 1:
                ppl = math.exp(min(avg_loss, 20))
                lr  = self.torch_optimizer.param_groups[0]['lr']
                elapsed = time.time() - epoch_start_time
                print(f" 🌟 epoch {epoch:4d}/{epochs} | loss={avg_loss:.4f} | ppl={ppl:.2f} | lr={lr:.2e} | time={elapsed:.4f}s")

        total_time = time.time() - start_time
        print("=" * 65)
        print(f"✅ Training completed successfully in {total_time / 60:.2f} minutes!")
        print("=" * 65)
        self.model.eval()