# =============================================================================
#  trainer.py  —  Training Engine  (PyTorch GPU + Fast Batching Version)
# =============================================================================

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim

# ─────────────────────────────────────────────────────────────────────────────
# Device setup
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = (
    torch.device("cuda")  if torch.cuda.is_available() else
    torch.device("mps")   if torch.backends.mps.is_available() else
    torch.device("cpu")
)

# ─────────────────────────────────────────────────────────────────────────────
# Old stubs for backward compatibility
# ─────────────────────────────────────────────────────────────────────────────
def cross_entropy_loss(logits: list, targets: list) -> tuple:
    return 0.0, []

class AdamOptimizer:
    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
        self.t = 0
    def step(self, params_and_grads: list) -> None:
        pass

def clip_gradients(params_and_grads: list, max_norm: float = 1.0) -> float:
    return max_norm

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP (FAST BATCHING)
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
        self.torch_optimizer.zero_grad()
        total_loss = 0.0
        
        # Batching: एक साथ कई sequence प्रोसेस करना
        for token_ids in batch_windows:
            if len(token_ids) < 2:
                continue
                
            inputs  = token_ids[:-1]
            targets = token_ids[1:]

            inp_t = torch.tensor(inputs,  dtype=torch.long,  device=DEVICE)
            tgt_t = torch.tensor(targets, dtype=torch.long,  device=DEVICE)

            with torch.enable_grad():
                ids     = inp_t
                seq_len = ids.shape[0]
                if seq_len > self.model.max_seq_len:
                    ids = ids[-self.model.max_seq_len:]
                    seq_len = self.model.max_seq_len

                pos    = torch.arange(seq_len, device=DEVICE)
                x      = self.model.token_emb(ids) + self.model.pos_emb(pos)
                for block in self.model.blocks:
                    x = block(x)
                x      = self.model.ln_final(x)
                logits = self.model.lm_head(x)

                loss = self._loss_fn(logits, tgt_t[:seq_len])
                loss = loss / len(batch_windows)
                loss.backward()

            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.torch_optimizer.step()
        return total_loss

    def train(
        self,
        all_token_ids: list,
        seq_len:       int,
        epochs:        int,
        log_every:     int = 10,
    ) -> None:
        
        windows = [all_token_ids[s : s + seq_len + 1] for s in range(0, len(all_token_ids) - seq_len, seq_len)]
        if not windows:
            print("[Trainer] Corpus too short.")
            return

        batch_size = 32  # 🚀 GPU को फ़ास्ट करने के लिए 32 का बैच
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.torch_optimizer,
            T_max=epochs,
            eta_min=1e-5,
        )

        print(f"[Trainer] Fast Batching Mode ON! device={DEVICE} | params={self.model.count_parameters():,}")
        self.model.train()

        for epoch in range(1, epochs + 1):
            random.shuffle(windows)
            batches = [windows[i:i + batch_size] for i in range(0, len(windows), batch_size)]
            
            epoch_loss = sum(self.train_step(b) for b in batches)
            avg_loss = epoch_loss / len(batches)
            
            self.loss_history.append(avg_loss)
            scheduler.step()

            if epoch % log_every == 0 or epoch == 1:
                ppl = math.exp(min(avg_loss, 20))
                lr  = self.torch_optimizer.param_groups[0]['lr']
                print(f"  epoch {epoch:4d}/{epochs}  |  loss={avg_loss:.4f}  |  ppl={ppl:.2f}  |  lr={lr:.2e}")

        print("[Trainer] Training complete.")
        self.model.eval()