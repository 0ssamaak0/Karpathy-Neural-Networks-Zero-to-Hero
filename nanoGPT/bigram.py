import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--batch_size', type=int, default=16, help='how many independent sequences will we process in parallel (default: 4)')
parser.add_argument('--block_size', type=int, default=32, help='what is the maximum context length for predictions? (default: 8)')
parser.add_argument('--max_iters', type=int, default=5000, help='how many training iterations do we want? (default: 3000)')
parser.add_argument('--eval_interval', type=int, default=100, help='how often do we evaluate the loss on train and val? (default: 300)')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='what is the learning rate for the optimizer? (default: 0.001)')
parser.add_argument('--eval_iters', type=int, default=200, help='how many batches we average the loss over for train and val (default: 200)')
parser.add_argument('--n_embed', type=int, default=64, help='how many dimensions do we want to embed the tokens in? (default: 32)')
parser.add_argument('--n_layer', type=int, default=4, help='how many layers of transformer blocks (default: 3)')
parser.add_argument("--n_head", type=int, default=4, help="how many heads do we want to use? (default: 4)")
parser.add_argument('--dropout', type=float, default=0.0, help='what dropout probability do we want to use? (default: 0.1)')

args = parser.parse_args()

# hyperparameters
batch_size = args.batch_size
block_size = args.block_size
max_iters = args.max_iters
eval_interval = args.eval_interval
learning_rate = args.learning_rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU if available
eval_iters = args.eval_iters
n_embed = args.n_embed
n_layer = args.n_layer
n_head = args.n_head
dropout = args.dropout

# reading the data
with open("../data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# all unique characters go here
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers and vice versa
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {val:key for key, val in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # Select the appropriate dataset based on the split parameter
    data = train_data if split == "train" else val_data

    # Generate a batch of random starting indices within the dataset
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Select a block of text of size block_size starting from each random index
    x = torch.stack([data[i:i+block_size] for i in ix])

    # Shift the selected block of text by one character to the right to create the target sequence
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)


@torch.no_grad()
# estimate the loss on train and val over a few batches
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    
# Single Head Attention Class
class Head(nn.Module):
    """ one head of self attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        # since tril isn't a parameter, we register it as a buffer
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # (B, T, C) @ (B, C, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0 , float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim = -1) # (B, T, T)
        wei = self.dropout(wei)

        # perform weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        return out

# Multi Head Attention Class
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        # linear transformation to the output of the multi-head attention as projection back to the residual pathway
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        # out is the outptu of the multi-head attention
        out =  torch.cat([h(x) for h in self.heads], dim = -1)
        # apply a linear layer to the concatenated output
        out = self.dropout(self.proj(out))
        return out

# Feed Forward Class
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            # multiply by 4 to follow the original implementation
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
# Transformer Block Class
class Block(nn.Module):
    """ Transformer Block: Communication followed by Computation """

    def __init__(self, n_embed, n_head):
        """ n_embed: embedding dimension
            n_head: number of heads in the multi-head attention
        """
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        # ln1 is applied directly on input before the multi-head attention
        self.ln1 = nn.LayerNorm(n_embed)
        # ln2 is applied directly on the output of the multi-head attention before the feed-forward layer
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        # residual connection (add the input to the output)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
# super simple bigram model
class BigramLanguageModel(nn.Module):
    # no need to pass vocab_size as an argument, since it is a global variable in this file
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a loockup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # each position is also associated with an embedding vector
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embed, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of ints
        token_emb = self.token_embedding_table(idx) # (B, T, C) = (4, 8 , vocab_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device = idx.device)) # (T, C) = (8, vocab_size)
        # x has the token identities + the position embeddings
        x = token_emb + pos_emb # (B, T, C) = (4, 8, vocab_size)
        # feed the input to the self attention head
        x = self.blocks(x) # (B, T, C) = (4, 8, vocab_size)
        # apply layer normalization
        x = self.ln_f(x) # (B, T, C) = (4, 8, vocab_size)
        logits = self.lm_head(x) # (B, T, vocab_size) = (4, 8, vocab_size)

        if targets is None:
            loss = None
        else:
            # note that F.cross_entropy accepts inputs in shape (B, C, T)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T) # can be as targets = targets.view(-1)
            
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] # (B, T)
            # get the logits for the next token
            logits, loss = self(idx_cond)
            # focus only on the last time step
            # (note that we are feeding the whole context each time, however we only care about the last prediction)
            # (this make doesn't make sense now, but the function will be modified later)
            logits = logits[:, -1, :] # Becomes (B, C) (get the last time step for each sequence)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim = -1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled token to the context
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T + 1)
        return idx
    

model = BigramLanguageModel().to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

# training loop
for i in range(max_iters):
    # every once in a while we evaluate the loss on train and val
    if i % eval_interval == 0:
        print(f" step {i} | train loss: {estimate_loss()['train']:.4f} | val loss: {estimate_loss()['val']:.4f}")
    # sample a batch of training data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()


# generate from the model
idx = torch.zeros((1,1), dtype = torch.long, device = device)
generated = model.generate(idx, 500) # shape (1, 101)
print(decode(generated[0].tolist()))
