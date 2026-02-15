"""
The most atomic way to train and inference a GPT in PyTorch.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import json
import math
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

docs = [l.strip()
        for l in open('../data/input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Tokenizer: unique characters in the dataset become token ids 0..n-1
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)  # token id for the special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1  # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")


@dataclass
class GPTConfig:
    n_embd: int = 16
    n_head: int = 4
    n_layer: int = 1
    block_size: int = 16


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
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
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(
            1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, T, T)
        attn = attn.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, nh, T, hs)
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


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MicroGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config)
                                    for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = GPTConfig()
model = MicroGPT(config).to(device)
print(f"num params: {sum(p.numel() for p in model.parameters())}")

learning_rate = 0.01
num_steps = 1000
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, betas=(0.85, 0.99), eps=1e-8)

loss_log = []
print("\n--- training ---")
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
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f} | {step_time:.4f}s")

train_time = time.time() - train_start
print(f"\ntotal training time: {train_time:.2f}s")

# Inference: may the model babble back to us
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
samples = []
model.eval()
with torch.no_grad():
    for sample_idx in range(20):
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
        sample = ''.join(uchars[i] for i in token_ids[1:])
        samples.append(sample)
        print(f"sample {sample_idx+1:2d}: {sample}")

# Save results for comparison notebook
results = {
    'loss_log': loss_log,
    'total_train_time': train_time,
    'samples': samples,
    'num_params': sum(p.numel() for p in model.parameters()),
}
with open('results_pytorch.json', 'w') as f:
    json.dump(results, f)
print(f"\nresults saved to results_pytorch.json")
