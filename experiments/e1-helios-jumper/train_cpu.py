#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CPU-only training script for experiment e1 (addition).
- Reads uint8 token streams from data/d1-focal-fossa/train.bin and val.bin
- Trains a decoder-only GPT-like transformer
- Saves best checkpoint to best-model.pth in the current experiment folder

Run:
  cd experiments/e1-helios-jumper
  python train_cpu.py --data_dir ../../data/d1-focal-fossa/ --out_dir .
"""

import os
import time
import math
import json
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def convert_seconds(s: float) -> str:
    s = int(s)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m > 0:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"

class Logger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.f = open(log_path, "a", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = f"[{now_str()}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


# -----------------------------
# Token batch loader
# -----------------------------

def get_batch(data: np.memmap, batch_size: int, block_size: int, device: str):
    # data is a 1D stream of token ids (uint8) in [0, vocab_size)
    # sample random contiguous chunks
    max_start = len(data) - (block_size + 1)
    if max_start <= 0:
        raise ValueError("Data too short for the given block_size.")

    ix = np.random.randint(0, max_start, size=(batch_size,))
    x = np.stack([data[i:i+block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i+1:i+1+block_size].astype(np.int64) for i in ix])

    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    return x, y


# -----------------------------
# Model (CPU-friendly)
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)

        # reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled dot-product attention (causal)
        # Use PyTorch SDPA if available; else fallback manual.
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # manual causal attention
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            if self.training and self.attn_dropout > 0:
                att = F.dropout(att, p=self.attn_dropout)
            y = att @ v  # (B, nh, T, hs)

        # back to (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int, n_embd: int, dropout: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            if targets is not None:
                targets = targets[:, -self.block_size:]
            B, T = idx.size()

        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# -----------------------------
# LR schedule
# -----------------------------

def get_lr(it: int, max_iters: int, base_lr: float, min_lr: float, warmup_iters: int):
    # warmup -> cosine decay
    if it < warmup_iters:
        return base_lr * (it + 1) / max(1, warmup_iters)
    if it >= max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, (max_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


# -----------------------------
# Main
# -----------------------------

@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    seed: int = 1337

    # model
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.0

    # train
    batch_size: int = 16
    eval_batch_size: int = 16
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 50

    # opt
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_frac: float = 0.1
    weight_decay: float = 0.1

    # misc
    vocab_size: int = 14  # e1 log shows 14


@torch.no_grad()
def estimate_loss(model: GPT, data: np.memmap, cfg: TrainConfig, device: str):
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch(data, cfg.eval_batch_size, cfg.block_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../data/d1-focal-fossa/", help="Folder containing train.bin/val.bin")
    parser.add_argument("--out_dir", type=str, default=".", help="Where to save best-model.pth and train_cpu.log")
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir if args.data_dir.endswith("/") else args.data_dir + "/",
        out_dir=args.out_dir,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    device = "cpu"
    set_seed(cfg.seed)

    ensure_dir(cfg.out_dir)
    log = Logger(os.path.join(cfg.out_dir, "train_cpu.log"))

    log.log("=== e1 CPU TRAIN START ===")
    log.log(f"device={device}")
    log.log(f"config={json.dumps(cfg.__dict__, indent=2)}")

    train_path = os.path.join(cfg.data_dir, "train.bin")
    val_path = os.path.join(cfg.data_dir, "val.bin")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing train.bin/val.bin in {cfg.data_dir}")

    train_data = np.memmap(train_path, dtype=np.uint8, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint8, mode="r")

    model = GPT(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay
    )

    warmup_iters = int(cfg.warmup_frac * cfg.max_iters)
    best_val = float("inf")
    train_start = time.time()

    # Evaluate before training
    t_loss = estimate_loss(model, train_data, cfg, device)
    v_loss = estimate_loss(model, val_data, cfg, device)
    log.log(f"init | train_loss={t_loss:.6f} | val_loss={v_loss:.6f}")

    for it in range(cfg.max_iters):
        lr = get_lr(it, cfg.max_iters, cfg.learning_rate, cfg.min_lr, warmup_iters)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_data, cfg.batch_size, cfg.block_size, device)
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (it + 1) % 50 == 0:
            elapsed = time.time() - train_start
            log.log(f"iter={it+1:6d}/{cfg.max_iters} | loss={loss.item():.6f} | lr={lr:.6g} | et={convert_seconds(elapsed)}")

        if (it + 1) % cfg.eval_interval == 0 or (it + 1) == cfg.max_iters:
            t_loss = estimate_loss(model, train_data, cfg, device)
            v_loss = estimate_loss(model, val_data, cfg, device)
            log.log(f"eval | iter={it+1} | train_loss={t_loss:.6f} | val_loss={v_loss:.6f}")

            if v_loss < best_val:
                best_val = v_loss
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "iter": it + 1,
                    "best_val_loss": best_val,
                }
                save_path = os.path.join(cfg.out_dir, "best-model.pth")
                torch.save(ckpt, save_path)
                log.log(f"saved best-model.pth (val_loss={best_val:.6f})")

    log.log("=== e1 CPU TRAIN DONE ===")
    log.close()


if __name__ == "__main__":
    main()
