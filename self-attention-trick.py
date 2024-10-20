import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1234)

B,T,C = 4,8,32

x = torch.randn(B,T,C)

# calculate the mean of past tokens using loops
'''xbow = torch.zeros(B,T,C)
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]
        xbow[b,t] = torch.mean(xprev, dim=0)'''

# calculate the mean of past tokens using matrix mul
'''weights = torch.tril(torch.ones(T,T))
weights = weights / weights.sum(dim=1, keepdim=True)
xbow2 = weights @ x'''

# calculate the mean of past tokens using softmax
'''tril = torch.tril(torch.ones(T,T))
wei = torch.zeros(T,T)
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei)
xbow3 = wei @ x

print(torch.allclose(xbow, xbow3))'''

# self attention
head_size = 16
query = nn.Linear(C, head_size, bias=False)
key = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # x (B,T,C) --> k (B,T,16)
q = query(x) # x (B,T,C) --> q (B,T,16)
v = value(x) # x (B,T,C) --> v (B,T,16)

wei = q @ k.transpose(2,1) / head_size**0.5 # (B,T,16) @ (B,16,T) --> (B,T,T)
tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril == 0, float('-inf')) # mask for self attention
wei = F.softmax(wei, dim=2)
xbow4 = wei @ v

print(xbow4.shape)