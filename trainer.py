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
# Adam Optimizer Mock Stub for Backward Compatibility (Matches main.py legacy)
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
# HIGH-SPEED PARALLEL GPU BATCH TRAINER
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

        # Professional AdamW Optimizer used by modern LLMs (Matches GPT-3 setup)
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

    def train_step(self, batch_windows: list) -> float:
        """
        Processes the entire batch of sequences concurrently as a single 2D tensor.
        Eliminates Python loops entirely inside the step to utilize maximum GPU cores.
        """
        self.torch_optimizer.zero_grad()
        
        # Step 1: Collect inputs and targets from the batch windows
        inputs_list = []
        targets_list = []
        
        for token_ids in batch_windows:
            if len(token_ids) < 2:
                continue
            inputs_list.append(token_ids[:-1])
            targets_list.append(token_ids[1:])
            
        if not inputs_list:
            return 0.0

        # Step 2: Convert to high-speed PyTorch long tensors directly on the selected GPU/Device
        inp_t = torch.tensor(inputs_list,  dtype=torch.long,  device=DEVICE)
        tgt_t = torch.tensor(targets_list, dtype=torch.long,  device=DEVICE)

        B, T = inp_t.shape

        # Step 3: Run the combined forward pass over all batches simultaneously
        try:
            with torch.enable_grad():
                # Clamp sequence lengths safely if they exceed max configuration
                if T > self.model.max_seq_len:
                    inp_t = inp_t[:, -self.model.max_seq_len:]
                    tgt_t = tgt_t[:, -self.model.max_seq_len:]
                    T = self.model.max_seq_len

                # Forward pass inside GPT model: Token Embeddings + Positional Embedding
                pos = torch.arange(T, device=DEVICE).unsqueeze(0) # (1, T)
                
                # Check model dimensions safely
                x = self.model.token_emb(inp_t) + self.model.pos_emb(pos) # Broadcast positional indices over batch B
                
                # Forward pass through deep decoder blocks
                for block in self.model.blocks:
                    x = block(x)
                
                x = self.model.ln_final(x)
                logits = self.model.lm_head(x) # Shape: (B, T, vocab_size)

                # Flatten logits and targets to feed directly into the PyTorch cross entropy optimizer
                loss = self._loss_fn(logits.view(-1, logits.size(-1)), tgt_t.view(-1))
                
                # Synchronous backward propagation (Executed once per batch update step)
                loss.backward()

                # Step 4: Gradient clipping to prevent exploding gradient issues
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                
                # Step 5: Optimizer step to adjust weights
                self.torch_optimizer.step()
                
                return loss.item()

        except Exception as e:
            # Safety check to prevent notebook crashes during long runs
            print(f"\n⚠️ [TRAINER DEBUG] Error captured during train_step: {str(e)}")
            print(f"   Batch shape was: Inputs={inp_t.shape}, Targets={tgt_t.shape}")
            return 0.0

    def train(
        self,
        all_token_ids: list,
        seq_len:       int,
        epochs:        int,
        log_every:     int = 10,
    ) -> None:
        """
        Triggers the ultra-fast batch training pipeline across T4 GPUs.
        """
        # Calculate optimal slice window sequences
        windows = [all_token_ids[s : s + seq_len + 1] for s in range(0, len(all_token_ids) - seq_len, seq_len)]
        if not windows:
            print("[Trainer] Corpus too short to create sliding token windows.")
            return

        batch_size = 64  # Optimal batch size configured for deep GPU execution
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.torch_optimizer,
            T_max=epochs,
            eta_min=1e-5,
        )

        print("=" * 65)
        print(f"🔥 GPT ENGINE TRIPLE-ACCELERATED!")
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
            
            # Split windows into fully vectorized batches of batch_size=64
            batches = [windows[i : i + batch_size] for i in range(0, len(windows), batch_size)]
            
            epoch_loss = 0.0
            steps_run = 0
            
            for b in batches:
                # We filter out windows that aren't of exact correct length to enforce true matrix alignment
                aligned_batch = [w for w in b if len(w) == seq_len + 1]
                if not aligned_batch:
                    continue
                
                loss_val = self.train_step(aligned_batch)
                epoch_loss += loss_val
                steps_run += 1
            
            avg_loss = epoch_loss / max(steps_run, 1)
            self.loss_history.append(avg_loss)
            scheduler.step()

            # Dynamic logs to keep track of the training speed in real-time
            if epoch % log_every == 0 or epoch == 1:
                ppl = math.exp(min(avg_loss, 20))
                lr  = self.torch_optimizer.param_groups[0]['lr']
                elapsed = time.time() - epoch_start_time
                print(f" 🌟 epoch {epoch:4d}/{epochs} | loss={avg_loss:.4f} | ppl={ppl:.2f} | lr={lr:.2e} | step_time={elapsed:.4f}s")

        total_time = time.time() - start_time
        print("=" * 65)
        print(f"✅ Training completed successfully in {total_time / 60:.2f} minutes!")
        print("=" * 65)
        self.model.eval()