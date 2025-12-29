#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CPU-only evaluation for experiment e1 (addition), compatible with the project's tokenizer order.

Run:
  cd experiments/e1-helios-jumper/_e1v1-super-tornado
  python eval_cpu.py --data_dir ../../../data/d1-focal-fossa/ --ckpt ../best-model.pth --limit 10000
"""

import os
import re
import csv
import time
import math
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import TinypyTokenizer  # <-- IMPORTANT: use the project's tokenizer


# -----------------------------
# Model (decoder-only GPT)
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
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
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v

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
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        return self.dropout(x)


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

    def forward(self, idx):
        B, T = idx.size()
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            B, T = idx.size()

        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# -----------------------------
# Generation + scoring
# -----------------------------

@torch.no_grad()
def greedy_generate_until_stop(model: GPT, idx: torch.Tensor, stop_id: int, max_new_tokens: int):
    # idx: (1, T)
    for _ in range(max_new_tokens):
        logits = model(idx)
        next_logits = logits[:, -1, :]          # (1, vocab)
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)  # (1,1)
        idx = torch.cat([idx, next_id], dim=1)
        if int(next_id.item()) == stop_id:
            break
    return idx

def strip_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../../data/d1-focal-fossa/", help="Folder containing test.txt")
    parser.add_argument("--ckpt", type=str, default="../best-model.pth", help="Path to best-model.pth")
    parser.add_argument("--limit", type=int, default=10000, help="Max number of test examples")
    parser.add_argument("--max_extra_tokens", type=int, default=5, help="Generate expected_len + extra tokens")
    args = parser.parse_args()

    device = "cpu"

    data_dir = args.data_dir if args.data_dir.endswith("/") else args.data_dir + "/"
    test_path = os.path.join(data_dir, "test.txt")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test.txt in {data_dir}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")

    # tokenizer (project's exact mapping)
    tokenizer = TinypyTokenizer()
    stop_id = tokenizer.encod_map["\n\n"]  # STOP token id used by dataset formatting

    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})

    vocab_size = int(cfg.get("vocab_size", len(tokenizer.keywords)))
    block_size = int(cfg.get("block_size", 128))
    n_layer = int(cfg.get("n_layer", 6))
    n_head = int(cfg.get("n_head", 8))
    n_embd = int(cfg.get("n_embd", 128))
    dropout = float(cfg.get("dropout", 0.0))

    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd, dropout).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # read + split examples exactly like the original eval.py
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = f.read()
    examples = test_data.split("\n\n")[:-1]
    examples = examples[: args.limit]

    os.makedirs("infers", exist_ok=True)
    successes_path = "infers/hard-match-successes.csv"
    failures_path = "infers/failures.csv"
    log_path = "model-eval_cpu.log"

    hard_ok = 0
    soft_ok = 0

    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] === e1 CPU EVAL START ===\n")
        lf.write(f"ckpt={args.ckpt}\n")
        lf.write(f"config={json.dumps({'vocab_size':vocab_size,'block_size':block_size,'n_layer':n_layer,'n_head':n_head,'n_embd':n_embd,'dropout':dropout}, indent=2)}\n")

    with open(successes_path, "w", newline="", encoding="utf-8") as sf, \
         open(failures_path, "w", newline="", encoding="utf-8") as ff:

        sw = csv.DictWriter(sf, fieldnames=["example_input", "example_output", "generated_output"])
        fw = csv.DictWriter(ff, fieldnames=["example_input", "example_output", "generated_output"])
        sw.writeheader()
        fw.writeheader()

        for i, ex in enumerate(examples, start=1):
            # parse prompt/output
            # ex is like: "#input\n...\n#output\n...\n" (no trailing \n\n because we split)
            if "#output\n" not in ex:
                continue
            prompt, expected = ex.split("#output\n", 1)
            prompt = prompt + "#output\n"

            prompt_ids = tokenizer.encode(prompt)
            expected_ids = tokenizer.encode(expected)

            x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

            max_new = len(expected_ids) + args.max_extra_tokens
            out = greedy_generate_until_stop(model, x, stop_id=stop_id, max_new_tokens=max_new)[0].tolist()

            gen_ids = out[len(prompt_ids):]

            # cut at first stop token if present
            if stop_id in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(stop_id)]

            generated = "".join(tokenizer.decod_map[t] for t in gen_ids)

            # hard token match
            if gen_ids == expected_ids:
                hard_ok += 1
                sw.writerow({"example_input": prompt, "example_output": expected, "generated_output": generated})
            else:
                # soft: ignore whitespace
                if strip_ws(generated) == strip_ws(expected):
                    soft_ok += 1
                fw.writerow({"example_input": prompt, "example_output": expected, "generated_output": generated})

    total = len(examples)
    hard_acc = 100.0 * hard_ok / max(1, total)
    soft_acc = 100.0 * soft_ok / max(1, total)

    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] |hard-accuracy: {hard_ok} = {hard_acc:.2f}% | soft-accuracy: {soft_ok} = {soft_acc:.2f}% |\n")
        lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] === e1 CPU EVAL DONE ===\n")

    print(f"Done. hard-accuracy: {hard_ok}/{total} = {hard_acc:.2f}% | soft-accuracy: {soft_ok}/{total} = {soft_acc:.2f}%")
    print(f"Wrote: {log_path}, {successes_path}, {failures_path}")


if __name__ == "__main__":
    main()
