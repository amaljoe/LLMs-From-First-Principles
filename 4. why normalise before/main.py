"""
Why normalise before: Pre-Norm vs Post-Norm vs No Normalization.

Step 3 uses Pre-Norm (the modern default): normalize BEFORE attention/MLP.
    x = x + Attn(Norm(x))
    x = x + MLP(Norm(x))

The original Transformer ("Attention is All You Need", Vaswani et al. 2017)
used Post-Norm: normalize AFTER the residual addition.
    x = Norm(x + Attn(x))
    x = Norm(x + MLP(x))

This script runs two experiments with the same hyperparameters as Step 3:
  1. Post-Norm  — original transformer style
  2. No-Norm   — normalization removed entirely

Reference: "On Layer Normalization in the Transformer Architecture"
  Xiong et al. 2020 — https://arxiv.org/abs/2002.04745
  Shows that Pre-Norm transformers don't need learning rate warmup and
  train more stably than Post-Norm.
"""

import json
import random
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Data loading (shared across experiments)
# ---------------------------------------------------------------------------
docs_raw = [l.strip()
            for l in open('../data/input.txt').read().strip().split('\n') if l.strip()]

uchars = sorted(set(''.join(docs_raw)))
BOS = len(uchars)
vocab_size = len(uchars) + 1


@dataclass
class GPTConfig:
    n_embd: int = 16
    n_head: int = 4
    n_layer: int = 8      # deeper than Step 3 to expose norm placement effects
    block_size: int = 16


# ---------------------------------------------------------------------------
# Shared modules (identical to Step 3)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(ms + self.eps)
        return x * self.scale


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(
            1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Pre-Norm Block: Norm BEFORE attention/MLP (modern default, used in Step 3)
#   x = x + Attn(Norm(x))
#   x = x + MLP(Norm(x))
# ---------------------------------------------------------------------------

class PreNormBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # pre-norm: norm BEFORE sub-layer
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Post-Norm Block: Norm AFTER the residual addition (original Transformer)
#   x = Norm(x + Attn(x))
#   x = Norm(x + MLP(x))
# ---------------------------------------------------------------------------

class PostNormBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln1 = RMSNorm(config.n_embd)
        self.ln2 = RMSNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.attn(x))   # post-norm: norm AFTER residual
        x = self.ln2(x + self.mlp(x))
        return x


# ---------------------------------------------------------------------------
# No-Norm Block: No normalization at all
#   x = x + Attn(x)
#   x = x + MLP(x)
# ---------------------------------------------------------------------------

class NoNormBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# MicroGPT with configurable norm placement
# ---------------------------------------------------------------------------

class MicroGPT(nn.Module):
    def __init__(self, config: GPTConfig, norm_type: str = 'pre'):
        super().__init__()
        self.config = config
        self.norm_type = norm_type
        self.wte = nn.Embedding(vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        block_cls = {'pre': PreNormBlock, 'post': PostNormBlock, 'none': NoNormBlock}
        if norm_type not in block_cls:
            raise ValueError(f"Unknown norm_type: {norm_type}")
        self.blocks = nn.ModuleList([block_cls[norm_type](config)
                                    for _ in range(config.n_layer)])

        # Pre-norm needs a final norm before lm_head; post-norm/none do not
        self.ln_f = RMSNorm(config.n_embd) if norm_type == 'pre' else None
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        if self.ln_f is not None:
            x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ---------------------------------------------------------------------------
# Training + inference loop (reused for each experiment)
# ---------------------------------------------------------------------------

def run_experiment(norm_type: str, num_steps: int = 1000):
    random.seed(42)
    torch.manual_seed(42)

    docs = list(docs_raw)
    random.shuffle(docs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPTConfig()
    model = MicroGPT(config, norm_type=norm_type).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  Experiment: {norm_type.upper()} normalization")
    print(f"  params: {n_params}")
    print(f"{'='*60}")

    learning_rate = 0.01
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.85, 0.99), eps=1e-8)

    loss_log = []
    train_start = time.time()

    for step in range(num_steps):
        step_start = time.time()

        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(config.block_size, len(tokens) - 1)

        x = torch.tensor(tokens[:n], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(tokens[1:n+1], dtype=torch.long,
                         device=device).unsqueeze(0)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        lr_t = learning_rate * (1 - step / num_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_t
        optimizer.step()

        step_time = time.time() - step_start
        loss_log.append({'step': step + 1, 'loss': loss.item(), 'time': step_time})
        if (step + 1) % 100 == 0 or step < 10:
            print(f"  step {step+1:4d}/{num_steps} | loss {loss.item():.4f}")

    train_time = time.time() - train_start
    print(f"  total training time: {train_time:.2f}s")

    # Inference
    temperature = 0.5
    samples = []
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            token_ids = [BOS]
            for _ in range(config.block_size):
                x = torch.tensor(token_ids, dtype=torch.long,
                                 device=device).unsqueeze(0)
                logits = model(x)
                logits = logits[:, -1, :] / max(temperature, 1e-6)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                if next_id == BOS:
                    break
                token_ids.append(next_id)
            samples.append(''.join(uchars[i] for i in token_ids[1:]))

    print(f"  samples: {', '.join(samples[:5])} ...")

    return {
        'loss_log': loss_log,
        'total_train_time': train_time,
        'samples': samples,
        'num_params': n_params,
        'norm_type': norm_type,
    }


# ---------------------------------------------------------------------------
# Run both experiments
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"num docs: {len(docs_raw)}")
    print(f"vocab size: {vocab_size}")

    results_pre = run_experiment('pre')
    results_post = run_experiment('post')
    results_none = run_experiment('none')

    with open('results_prenorm.json', 'w') as f:
        json.dump(results_pre, f)
    with open('results_postnorm.json', 'w') as f:
        json.dump(results_post, f)
    with open('results_nonorm.json', 'w') as f:
        json.dump(results_none, f)

    print("\n--- results saved ---")
    print("  results_prenorm.json")
    print("  results_postnorm.json")
    print("  results_nonorm.json")
