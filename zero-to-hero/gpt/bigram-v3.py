''' Now we're going to be implementing LayerNorm into our model, as well as some other stuff 

This is the final version of the bigram model before we are really starting to scale it up

'''

import torch
import torch.nn as nn
from torch.nn import functional as F

### Instantiate hyperparameters
batch_size = 32 # number of independent sequences to be parallel processed
block_size = 8 # maximum context length for predictions 
max_iters = 5000 # maximum number of iterations
eval_interval = 300 # every epoch interval (I'm guessing)
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu' # for when I'm not GPU poor
eval_iters = 200
n_embd = 32
# ------------------- #

torch.manual_seed(1337)
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# we already downloaded this, so no need to import
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# set up all characters, and encode/decode information
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: takes in a string, outputs integer list
decode = lambda l: ''.join([itos[i] for i in l]) # input integer list, output string

# define dataset and split accordingly
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
   
    # create inputs and target data
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    ''' one head of self attention --- is the communication part of the transformer '''
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        # create keys and values via x @ W^T for W = key, query
        k = self.key(x)
        q = self.query(x)
        # compute attention scores, i.e. affinities
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) --> (B, T, T), and preserving variance
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # creating the decoder block, i.e. shielding the past
        wei = F.softmax(wei, dim=-1)
        # now perform the weighted aggregation
        v = self.value(x)
        out = wei @ v
        return out


class MultiAttentionHead(nn.Module):
    ''' multiple heads of self attention that run in parallel 
    
    We'll be adding the skip connections here now, which we're adding as self.proj
    
    '''
    ## this really helps the model gather more context about what the letters are doing, and how they're interacting with each other
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # creates multiple heads that will work in parallel
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # do the forward pass in the channel, i.e. vocab size, dimension
        out = self.proj(out)
        return out
        

class FeedForward(nn.Module):
    ''' a linear layer followed by a non-linearity, the computation part of the transformer '''
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(n_embd,  4 * n_embd), # factor of 4 comes from the paper, Sec. 3.3, alternating for the projection
                nn.ReLU(),
                nn.Linear(4* n_embd, n_embd), # the self projection layer, i.e. the skip connection layer
            )
        
    def forward(self, x):
        return self.net(x)
    


class Block(nn.Module):
    ''' Blocks will help chunk the learning process into communication (attention) and computation (feed forward) 
    Now we'll also want to add the skip connections. These allow the gradients to flow through, since addition will just
    transfer the gradients without any kind of updating. It will help the transformer in the optimization process as the training
    continues to progress. These skip the residual stream. 
    
    '''
    def __init__(self, n_embd, n_head):
        # n_embd = embedding dimension
        # n_head = number of heads we want in the block
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiAttentionHead(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # now we'll incorporate the LayerNorm in our block, slightly deviating from the paper though (pre-layer norm it's called)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x): 
        # the purpose of adding x = x + self.sa(x) is because the sa goes off and does some communication, then comes back
        # meaning that during the computation part --- feed forward --- we're implementing a communication mechanism
        # based on what we're finding during the computation process
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

# THE SIMPLE BIGRAM LANGUAGE MODEL -- THANKS ANDREJ
# we utilize embeddings in order to avoid sparsity, i.e. one hot vectors
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # each token will utilize the logits to predict the next token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        ## now create the gpt in terms of blocks so that we have both communication and computation
        # having multiple blocks allows the transformer to communicate multiple times to each other
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd)
        )

        # create the language model layer -- I deleted the previous layers that were originally here in earlier versions
        self.lm_head = nn.Linear(n_embd, vocab_size)
        

    # (batch, time, channel) = (4, 8, 65)
    def forward(self, idx, targets=None):
        B, T = idx.shape    

        # create a token embedding, which is their identity 
        tok_emb = self.token_embedding_table(idx) # (batch, time, channel)

        # create a position embedding, which keeps track of their positions
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)

        # create input
        x = tok_emb + pos_emb # becomes (B, T, C) after right aligning and tensor broadcasting
        # do the multi self attention heads on the input now
        #x = self.sa_heads(x)
        # then implement the feed forward layer on the inputs, allowing the model to "think" about the inputs more
        #x = self.ffwd(x)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None            
        else:
            # need to reshape logits so that it fits into cross entropy functional form
            # bc pytorch expects you to pass in your logits this way
            B, T, C = logits.shape
            logits = logits.view(B*T, C)

            # need to do the same as targets
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    
    # now let's generate from the trained model
    # idx is the current context of some characters in a batch -- (B, T)
    # generate extends it to be (B, T+1), then again to (B, T+2) ...
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # need to crop idx in order to preserve scope throughout the iteration, otherwise you'll run into errors
            idx_cond = idx[:, -block_size:]

            # now get logits and loss based on these conditions on the indices of x
            logits, loss = self(idx_cond)

            # for now we'll only focus on the last time step (last part of the block)
            # since the predictions are following whatever was there previously
            logits = logits[:, -1, :] # becomes (B, C)
            
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1) # dim = (B, C)
            
            # sample from the distribution --- one sample per batch
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to a running sequence of indices throughout gen process
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx

# instantiate my bigram language model
model = BigramLanguageModel()
m = model.to(device)

# create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# the training loop
for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print('------------')
print("training done, generating data:")
print('------------')
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=200)[0].tolist()))
