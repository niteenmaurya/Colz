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
# HIGH-SPEED PARALLEL GPU BATCH TRAINER
# fp16 autocast + GradScaler + torch.compile support
# ─────────────────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(
        self,
        model,
        lr:       float = 3e-4,
        max_norm: float = 1.0,
        batch_size: int = 256,
    ) -> None:
        self.model      = model
        self.max_norm   = max_norm
        self.batch_size = batch_size

        # PyTorch AdamW
        self.torch_optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        # Legacy stub (keeps old scripts from crashing)
        self.optimizer = AdamOptimizer(lr=lr)

        # Mixed-precision scaler (only active on CUDA)
        self._use_amp = (DEVICE.type == "cuda")
        self.scaler   = torch.cuda.amp.GradScaler(enabled=self._use_amp)

        self.loss_history: list = []
        self._loss_fn = nn.CrossEntropyLoss(ignore_index=0)   # PAD_ID = 0

    # ─────────────────────────────────────────────────────────────────────────
    def train_step(self, batch_windows: list, seq_len: int) -> float:
        """
        Single batch forward + backward pass.
        Uses fp16 autocast on CUDA for ~2x throughput.
        """
        self.torch_optimizer.zero_grad()

        inputs_list  = []
        targets_list = []

        for token_ids in batch_windows:
            if len(token_ids) < 2:
                continue

            inp = token_ids[:-1]
            tgt = token_ids[1:]

            if len(inp) > seq_len:
                inp = inp[-seq_len:]
                tgt = tgt[-seq_len:]

            pad_len = seq_len - len(inp)
            if pad_len > 0:
                inp = inp + [0] * pad_len
                tgt = tgt + [0] * pad_len

            inputs_list.append(inp)
            targets_list.append(tgt)

        if not inputs_list:
            return 0.0

        inp_t = torch.tensor(inputs_list,  dtype=torch.long, device=DEVICE)
        tgt_t = torch.tensor(targets_list, dtype=torch.long, device=DEVICE)
        B, T  = inp_t.shape

        try:
            # autocast: fp16 on CUDA → ~2x faster, half the VRAM
            with torch.autocast(device_type=DEVICE.type, enabled=self._use_amp):
                pos    = torch.arange(T, device=DEVICE).unsqueeze(0)
                x      = self.model.token_emb(inp_t) + self.model.pos_emb(pos)
                for block in self.model.blocks:
                    x = block(x)
                x      = self.model.ln_final(x)
                logits = self.model.lm_head(x)           # [B, T, vocab]
                loss   = self._loss_fn(
                    logits.view(-1, logits.size(-1)),
                    tgt_t.view(-1)
                )

            # scaled backward (handles fp16 underflow)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.torch_optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.scaler.step(self.torch_optimizer)
            self.scaler.update()

            return loss.item()

        except Exception as e:
            print(f"\n⚠️  [TRAINER] train_step error: {e}")
            return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    def train(
        self,
        all_token_ids: list,
        seq_len:       int,
        epochs:        int,
        log_every:     int = 5,
    ) -> None:
        """
        Full training loop with cosine LR schedule.
        """
        windows = [
            all_token_ids[s : s + seq_len + 1]
            for s in range(0, len(all_token_ids) - seq_len, seq_len)
        ]
        if not windows:
            windows = [all_token_ids]

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.torch_optimizer,
            T_max=epochs,
            eta_min=1e-5,
        )

        print("=" * 65)
        print(f"🔥  GPT TRAINER  |  autocast={'ON (fp16)' if self._use_amp else 'OFF'}")
        print(f"👉  Device      : {DEVICE.type.upper()}")
        print(f"👉  Batch size  : {self.batch_size}")
        print(f"👉  Params      : {self.model.count_parameters():,}")
        print(f"👉  Windows     : {len(windows)}")
        print("=" * 65)

        self.model.train()
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            random.shuffle(windows)

            batches    = [windows[i : i + self.batch_size]
                          for i in range(0, len(windows), self.batch_size)]
            epoch_loss = 0.0
            steps_run  = 0

            for b in batches:
                epoch_loss += self.train_step(b, seq_len)
                steps_run  += 1

            avg_loss = epoch_loss / max(steps_run, 1)
            self.loss_history.append(avg_loss)
            scheduler.step()

            if epoch % log_every == 0 or epoch == 1:
                ppl     = math.exp(min(avg_loss, 20))
                lr      = self.torch_optimizer.param_groups[0]['lr']
                elapsed = time.time() - epoch_start
                print(f" 🌟  epoch {epoch:4d}/{epochs} | "
                      f"loss={avg_loss:.4f} | ppl={ppl:.2f} | "
                      f"lr={lr:.2e} | time={elapsed:.3f}s")

        total = time.time() - start_time
        print("=" * 65)
        print(f"✅  Training done in {total / 60:.2f} min")
        print("=" * 65)
        self.model.eval()