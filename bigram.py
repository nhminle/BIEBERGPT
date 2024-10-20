import torch
from torch.nn import functional as F
import torch.nn as nn
torch.manual_seed(69420)

with open('Bieber-bible.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# define hyperparam
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
block_size = 8 # context length for prediction
batch_size = 32 # number of seq to process in parallel
learning_rate = 1e-3 
training_step = 10000 
eval_iters = 100
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

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
    

class BigramLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are (B,T) aka batch x time tensor
        logits = self.token_embedding_table(idx) # (B,T,C) batch size x block size x vocab size
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
            logits, loss = self(idx) # logits -> (B, T, C)
            # get the last token of the batch 
            logits = logits[:, -1,:] # -> (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) 
            # sample from the distribution to get the predicted next token 
            idx_next = torch.multinomial(probs, num_samples=1) # -> (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # -> (B, T+1)
        return idx

model = BigramLM().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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
print((decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist())))