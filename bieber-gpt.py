import torch
from torch.nn import functional as F
import torch.nn as nn
torch.manual_seed(69420)

with open('Bieber-bible.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# define hyperparam
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
block_size = 128 # context length for prediction
batch_size = 32 # number of seq to process in parallel
learning_rate = 3e-4 
training_step = 5000 
eval_iters = 100
eval_interval = 500
embed_dim = 128
n_layers = 6 # number of transformer blocks in the model
n_heads = 4 # number of heads for self attention
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'device: {device}')

# mapping from vocab to number
stoi = {chr:i for i, chr in enumerate(vocab)}
itos = {i:chr for i, chr in enumerate(vocab)}

encode = lambda s: [stoi[c] for c in s] # input: string, output: list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # input: list of ints, output: string

# encode text and store in a tensor
data = torch.tensor(encode(text), dtype=torch.long)

# split data to training and validation
split = int(len(data) * 0.9)
training_data = data[:split]
val_data = data[split:]

def get_batch(split_type):
    """
    create a batch from data

    Args:
        split_type (str): 'train' or 'val'.

    Returns:
        x (tensor): tensor containing the context.
        y (tensor): tensor containing ground truth label.
    """
    data = training_data if split_type == 'train' else val_data
    indx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in indx])
    y = torch.stack([data[i+1:i+1+block_size] for i in indx])
    return x.to(device), y.to(device)

@torch.no_grad
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x,y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    return out

class FeedForward(nn.Module):
    """
    simple linear layer followed by RelU
    idea: try 2 linear transf with a RelU in between

    Args:
        emb_dim (int): embedding dimension.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*4),
            nn.ReLU(),
            nn.Linear(emb_dim*4, emb_dim),
            nn.Dropout(dropout) # dropout 
        )

    def forward(self, x):
        return self.ff(x)

class Head(nn.Module):
    """
    single head for self attention

    Args:
        head_size (int): size of head.
    """
    def __init__(self, head_size) -> None:
        super().__init__()
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # x (B,T,C) --> k (B,T,head_size)
        q = self.query(x) # x (B,T,C) --> q (B,T,head_size)
        v = self.value(x) # x (B,T,C) --> v (B,T,head_size)
        # calculate attention score 
        weights = q @ k.transpose(2,1) * C**-0.5 # (B,T,C) @ (B,C,T) --> (B,T,T)
        weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        weights = F.softmax(weights, dim=-1) # (B,T,T)
        weights = self.dropout(weights)
        out = weights @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out

class MultiHead(nn.Module):
    """
    multi head for self attention

    Args:
        num_head (int): number of heads
        head_size (int): size of head.
    """
    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    """
    Transformer block: communicate then compute

    Args:
        embed_dim (int): embedding dimension.
        num_head (int): number of heads
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_size = embed_dim // num_heads
        self.sat_heads = MultiHead(num_heads, head_size)
        self.ffwd = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.sat_heads(self.ln1(x)) # += to create residual connections, & pre norm formulation
        x = x + self.ffwd(self.ln2(x)) # += to create residual connections, & pre norm formulation
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        # self.sat_head = Head(embed_dim)
        # self.sat_heads = MultiHead(4, embed_dim //4) # aka 4 heads of 8 dimensional self attention
        # self.ffwd = FeedForward(embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads=n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, idx, targets=None):
        B, T = idx.shape 

        # idx and targets are (B,T) aka batch x time tensor
        tok_emb = self.token_embedding_table(idx) # (batch size x block size x embedding dimension)
        loc_emb = self.position_embedding_table(torch.arange(T, device=device)) # (block size x embedding dimension)
        x = tok_emb + loc_emb # (B x T x embedding dimension)

        # x = self.sat_heads(x) # apply multi head self attention
        # x = self.ffwd(x) # (B x T x embedding dimension)

        x = self.blocks(x) # (B x T x embedding dimension)
        x = self.ln(x) # (B x T x embedding dimension)
        logits = self.lm_head(x) # (B,T, vocab_size)

        loss = None # loss default to none
        if targets is not None:
            # reshape logits to calculate cross_entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the idx to fit block size
            idx_cont = idx[:, -block_size:]
            logits, loss = self(idx_cont) # logits -> (B, T, C)
            # get the last token of the batch 
            logits = logits[:, -1,:] # -> (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) 
            # sample from the distribution to get the predicted next token 
            idx_next = torch.multinomial(probs, num_samples=1) # -> (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # -> (B, T+1)
        return idx

model = GPT().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(training_step):
    # eval loss every eval_interval
    if epoch % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print((decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=600)[0].tolist())))