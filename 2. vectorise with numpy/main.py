"""
Vectorise with NumPy: replacing Python scalar loops with numpy array operations.
Same GPT architecture as 1. microgpt, but dramatically faster via vectorized math.
The autograd engine now operates on numpy arrays instead of individual Python floats.

@karpathy (original), vectorized adaptation
"""

import json
import time
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# --- Data Loading ---
docs = [l.strip() for l in open('../data/input.txt').read().strip().split('\n')
        if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# --- Tensor Autograd Engine (operates on numpy arrays) ---


class Tensor:
    """A numpy-backed tensor with autograd support (reverse-mode autodiff)."""

    def __init__(self, data, _children=(), _op=''):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._backward = lambda: None
        self._children = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            # handle broadcasting: sum over axes that were broadcast
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            other.grad += _unbroadcast(-self.data / (other.data ** 2) * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self

    def matmul(self, other):
        """Matrix multiply: self @ other."""
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            if self.data.ndim == 1 and other.data.ndim == 2:
                # (D,) @ (D, M) -> (M,)
                self.grad += out.grad @ other.data.T
                other.grad += np.outer(self.data, out.grad)
            elif self.data.ndim == 2 and other.data.ndim == 1:
                # (N, D) @ (D,) -> (N,)
                self.grad += np.outer(out.grad, other.data)
                other.grad += self.data.T @ out.grad
            else:
                # (N, D) @ (D, M) -> (N, M)
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        out = Tensor(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0).astype(np.float64) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            if axis is None:
                self.grad += np.ones_like(self.data) * out.grad
            else:
                grad = out.grad
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.data.shape)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / n

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), 'getitem')

        def _backward():
            np.add.at(self.grad, idx, out.grad)
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data, dtype=np.float64)
        for v in reversed(topo):
            v._backward()


def _unbroadcast(grad, shape):
    """Sum out dimensions that were broadcast to match target shape."""
    if grad.shape == shape:
        return grad
    # scalar target
    if shape == ():
        return np.sum(grad)
    # sum leading dims that don't exist in target
    ndim_diff = grad.ndim - len(shape)
    for _ in range(ndim_diff):
        grad = grad.sum(axis=0)
    # sum dims of size 1 in target
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


# --- Model Hyperparameters (same as 1. microgpt) ---
n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
head_dim = n_embd // n_head

# --- Parameter Initialization ---
std = 0.08


def param(shape):
    return Tensor(np.random.randn(*shape) * std)


state_dict = {
    'wte': param((vocab_size, n_embd)),
    'wpe': param((block_size, n_embd)),
    'lm_head': param((vocab_size, n_embd)),
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = param((n_embd, n_embd))
    state_dict[f'layer{i}.attn_wk'] = param((n_embd, n_embd))
    state_dict[f'layer{i}.attn_wv'] = param((n_embd, n_embd))
    state_dict[f'layer{i}.attn_wo'] = param((n_embd, n_embd))
    state_dict[f'layer{i}.mlp_fc1'] = param((4 * n_embd, n_embd))
    state_dict[f'layer{i}.mlp_fc2'] = param((n_embd, 4 * n_embd))

params = list(state_dict.values())
print(f"num params: {sum(p.data.size for p in params)}")

# --- Model Functions (vectorized with numpy) ---


def rmsnorm(x):
    """x: Tensor of shape (D,)"""
    ms = (x * x).mean()
    scale = (ms + 1e-5) ** -0.5
    return x * scale


def softmax(logits):
    """logits: Tensor of shape (V,) or (T,)"""
    max_val = np.max(logits.data)
    shifted = logits - Tensor(max_val)
    exps = shifted.exp()
    total = exps.sum()
    return exps / total


def gpt(token_id, pos_id, keys, values):
    """Forward one token through the GPT, using KV cache."""
    # Embeddings: just index into the weight matrices (vectorized lookup)
    tok_emb = state_dict['wte'][token_id]  # (n_embd,)
    pos_emb = state_dict['wpe'][pos_id]    # (n_embd,)
    x = tok_emb + pos_emb
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x

        # Pre-norm
        x = rmsnorm(x)

        # Q, K, V projections: single matmul each instead of list comprehension
        wq = state_dict[f'layer{li}.attn_wq']
        wk = state_dict[f'layer{li}.attn_wk']
        wv = state_dict[f'layer{li}.attn_wv']
        wo = state_dict[f'layer{li}.attn_wo']

        q = linear(x, wq)  # (n_embd,)
        k = linear(x, wk)
        v = linear(x, wv)

        keys[li].append(k)
        values[li].append(v)

        # Multi-head attention
        x_attn_heads = []
        for h in range(n_head):
            hs = h * head_dim
            he = hs + head_dim

            q_h = q[hs:he]  # (head_dim,)

            # Stack all cached keys and values for this head
            n_cached = len(keys[li])
            # Build attention logits for all cached positions
            attn_logits_list = []
            for t in range(n_cached):
                k_t = keys[li][t][hs:he]  # (head_dim,)
                # dot product: sum of element-wise multiply
                dot = (q_h * k_t).sum() / (head_dim ** 0.5)
                attn_logits_list.append(dot)

            # Stack into a single tensor for softmax
            attn_data = np.array([a.data for a in attn_logits_list])
            attn_logits = Tensor(attn_data, tuple(attn_logits_list), 'stack')

            def make_stack_backward(logits_list, stacked):
                def _backward():
                    for t_idx, a in enumerate(logits_list):
                        a.grad += stacked.grad[t_idx]
                return _backward
            attn_logits._backward = make_stack_backward(attn_logits_list, attn_logits)

            attn_weights = softmax(attn_logits)  # (n_cached,)

            # Weighted sum of values
            head_out_data = np.zeros(head_dim)
            v_heads = [values[li][t][hs:he] for t in range(n_cached)]
            for t in range(n_cached):
                head_out_data += attn_weights.data[t] * v_heads[t].data

            head_out = Tensor(head_out_data, (attn_weights, *v_heads), 'attn_combine')

            def make_combine_backward(weights, v_list, h_out, hd):
                def _backward():
                    for t_idx in range(len(v_list)):
                        # grad w.r.t. attn_weights[t]
                        weights.grad[t_idx] += np.dot(v_list[t_idx].data, h_out.grad)
                        # grad w.r.t. v[t]
                        v_list[t_idx].grad += weights.data[t_idx] * h_out.grad
                return _backward
            head_out._backward = make_combine_backward(attn_weights, v_heads, head_out, head_dim)

            x_attn_heads.append(head_out)

        # Concatenate heads
        concat_data = np.concatenate([h.data for h in x_attn_heads])
        x_attn = Tensor(concat_data, tuple(x_attn_heads), 'concat')

        def make_concat_backward(heads, concatenated, hdim):
            def _backward():
                for i_h, h in enumerate(heads):
                    h.grad += concatenated.grad[i_h * hdim:(i_h + 1) * hdim]
            return _backward
        x_attn._backward = make_concat_backward(x_attn_heads, x_attn, head_dim)

        # Output projection
        x = linear(x_attn, wo)
        x = x + x_residual

        # MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = x.relu()
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = x + x_residual

    logits = linear(x, state_dict['lm_head'])
    return logits


def linear(x, w):
    """x: Tensor (D,), w: Tensor (Dout, Din) -> Tensor (Dout,)"""
    # out = W @ x
    out_data = w.data @ x.data
    out = Tensor(out_data, (x, w), 'linear')

    def _backward():
        # dL/dx = W^T @ dL/dout
        x.grad += w.data.T @ out.grad
        # dL/dW = dL/dout (outer) x
        w.grad += np.outer(out.grad, x.data)
    out._backward = _backward
    return out


# --- Training ---
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_buf = [np.zeros_like(p.data) for p in params]
v_buf = [np.zeros_like(p.data) for p in params]

num_steps = 1000
loss_log = []

print("\n--- training ---")
train_start = time.time()

for step in range(num_steps):
    step_start = time.time()

    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys_cache = [[] for _ in range(n_layer)]
    values_cache = [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys_cache, values_cache)
        probs = softmax(logits)
        loss_t = -(probs[target_id]).log()
        losses.append(loss_t)

    # Average loss
    loss_sum_data = sum(l.data for l in losses)
    loss_val = loss_sum_data / n
    # Build loss node
    loss = Tensor(loss_val, tuple(losses), 'mean_loss')

    def make_mean_backward(loss_list, loss_node, n_tokens):
        def _backward():
            for l in loss_list:
                l.grad += loss_node.grad / n_tokens
        return _backward
    loss._backward = make_mean_backward(losses, loss, n)

    # Backward
    loss.backward()

    # Adam update (vectorized on arrays)
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_buf[i] / (1 - beta1 ** (step + 1))
        v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (np.sqrt(v_hat) + eps_adam)
        p.grad = np.zeros_like(p.data)

    step_time = time.time() - step_start
    loss_log.append({'step': step + 1, 'loss': float(loss.data), 'time': step_time})
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f} | {step_time:.3f}s")

train_time = time.time() - train_start
print(f"\ntotal training time: {train_time:.2f}s")

# --- Inference ---
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
samples = []
for sample_idx in range(20):
    keys_cache = [[] for _ in range(n_layer)]
    values_cache = [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys_cache, values_cache)
        probs = softmax(logits / temperature)
        token_id = np.random.choice(vocab_size, p=probs.data)
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    name = ''.join(sample)
    samples.append(name)
    print(f"sample {sample_idx+1:2d}: {name}")

# --- Save results for comparison notebook ---
results = {
    'loss_log': loss_log,
    'total_train_time': train_time,
    'samples': samples,
    'num_params': sum(p.data.size for p in params),
}
with open('results_numpy.json', 'w') as f:
    json.dump(results, f)
print(f"\nresults saved to results_numpy.json")
