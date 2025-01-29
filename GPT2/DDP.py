import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time
import math

from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# torchrun command sets the env vars RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a distributed run or not
if ddp:
    # you should be using cuda
    assert torch.cuda.is_available()
    init_process_group("nccl")  # initialize the process group
    ddp_rank = int(
        os.environ["RANK"]
    )  # GPU 0 have rank of 0, GPU 1 have rank of 1, etc.
    ddp_local_rank = int(
        os.environ["LOCAL_RANK"]
    )  # Rank of the GPU on the node (we have a single node)
    ddp_world_size = int(os.environ["WORLD_SIZE"])  # world size is the number of GPUs
    device = torch.device(f"cuda:{ddp_local_rank}")
    torch.cuda.set_device(device)
    master_process = (
        ddp_rank == 0
    )  # process 0 some additional work. It's responsible for logging, saving checkpoints, etc.
else:
    # single GPU
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = "cuda"
    master_process = True # single GPU, so we are the master process

total_batch_size = 2**19
B = 16  # micro batch size
T = 1024  # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0  # total batch size must be divisible by B * T * world size
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # divide by world size to distribute the batch over all GPUs
# print only if master process
if master_process:
    print(
        f"total desired batch size: {total_batch_size:,}, grad_accum_steps: {grad_accum_steps:,}"
    )

# # add some logging
# print(f"I'm GPU: {ddp_rank}\nBye")

# Modified Data loader
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        # Process rank 0 will start at 0
        # Process rank 1 will start at B * T
        # Process rank 2 will start at 2 * B * T 
        # etc.
        self.current_position = self.B * self.T * self.process_rank 

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + (B * T) + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        # reset if we reach the end
        if self.current_position + (B * T * self.num_processes) + 1 > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y
    
train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size)

# GPT Model class (no changes)
@dataclass
class GPTConfig:
    block_size: int = 1024  # maximum sequence length
    vocab_size: int = (
        50257  # number of tokens (50k BPE merges + 256 byte tokens + 1 <|endoftext|> token)
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding size


# Multi-Head Attention (in a single class)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Bias (or mask)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        # batch size, sequence length, embedding size (n_embd)
        B, T, C = x.size()
        # Query, Key, Value (extract them from c_attn)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        # n_head is treated as a batch dimension
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, n_head, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, n_head, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, n_head, T, hs)
        # # Attention (Comment this since we will use flash attention)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / (C // self.n_head) ** 0.5)
        # # apply the mask
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # # apply the softmax
        # att = F.softmax(att, dim=-1)
        # apply the attention
        # y = att @ v

        # Flash attention (torch.compile will compile this into flash attention)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # transpose and reshape
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                # token embedding
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                # positional embedding
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                # transformer layers
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # final layer norm (Before the Linear layer)
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            # 1 / sqrt(2 * number of residual layers) note that each layer has two residual connections
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        # Shape of idx is (B, T) (Batch size, Sequence length)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Can't forward a sequence of length {T} longer than the block size of {self.config.block_size}"
        # Get the token embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # Shape is (T)
        pos_emb = self.transformer.wpe(pos)  # Shape is (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # Shape is (B, T, n_embd)
        x = tok_emb + pos_emb  # Shape is (B, T, n_embd) Broadcasting in addition
        # Forward pass through the transformer layers
        for block in self.transformer.h:
            x = block(x)
        # Final layer norm
        x = self.transformer.ln_f(x)
        # Get the logits
        logits = self.lm_head(x)  # Shape is (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )  # (B * T, vocab_size)
        return logits, loss

    # The goal of the function to separate the parameters (should be weight decayed and not weight decayed)
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all params requiring grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, others won't
        # all weights in matmuls and embeddings will be weight decayed
        # biases and layernorms won't be weight decayed
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_dcay_params = sum(p.numel() for p in decay_params)
        num_ndcay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(
                f"num ecayed parameter tensors: {len(decay_params)} with {num_dcay_params:,} params"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)} with {num_ndcay_params:,} params"
            )
        fused = "cuda" in device
        # fused make it faster
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused
        )
        return optimizer

# Set manual seed
# Set fixed seed (to create identical models for all processes)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# initialize the model as before
model = GPT(GPTConfig(vocab_size = 50304)) 
model = model.to(device)
model = torch.compile(model)
# if ddp you must wrap the model in DDP
if ddp:
    model = DDP(model, device_ids = [ddp_local_rank]) # To combine the gradients from all GPUs into one


# LR schedule (no changes)
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# Training loop
times = []
train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size)
if ddp:
    optimizer = model.module.configure_optimizers(0.1, 6e-4, "cuda")
else:
    optimizer = model.configure_optimizers(0.1, 6e-4, "cuda")
losses = []
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to("cuda"), y.to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss /= (
            grad_accum_steps  # divide by grad_accum_steps since the reduction is mean
        )
        loss_accum += loss.detach()
        # The loss is synchronized across all GPUs for each microsetp. This is wasteful.
        if ddp:
            # equivalent to using `no_sync` context manager
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # turn on only if it's the last microstep
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)  # avg the loss across all GPUs
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set lr for the current iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    losses.append(loss_accum)
    if master_process:
        print(
            f"Step {step}, Loss: {loss_accum}, lr: {lr:.4e}, norm: {norm:.4f}, Time: {(t1 - t0 )* 1000:.3f}ms"
        )
    times.append(t1 - t0)


# Destroy the process group
if ddp:
    destroy_process_group()