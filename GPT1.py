import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken 
from torch import save, load

print(torch.cuda.is_available())
# Hyperparameters

torch.manual_seed(1337)

batch_size = 64
block_size = 256
n_embd = 126
max_iters = 3000
eval_interval = 200
learning_rate = 2e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
encoder = 'tiktoken' 
n_head = 6
n_layers = 6
Dropout = 0.2  # Adjust dropout rate as needed

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

if encoder == 'tiktoken':
  import tiktoken
  enc = tiktoken.get_encoding("o200k_base")
  assert enc.decode(enc.encode("hello world")) == "hello world"
  vocab_size = 50257


  data = torch.tensor(enc.encode(text), dtype=torch.long)

elif encoder == 'nltk':

  import nltk
  from nltk.corpus import words
  
  nltk.download('words')
  
  english_words = set(words.words())

  print(len(english_words), "English words loaded.")
  
  vocab_size = len(english_words) # +1 for unknown words
  
  tokens = word_tokenize(text)
  data = torch.tensor([ord(c) for c in tokens], dtype=torch.long)

elif encoder == 'base':
    with open('input.txt', 'r', encoding='utf-8') as f:
       text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch : i for i, ch in enumerate(chars)}
    itos = {i : ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype= torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# TODO :: Try and train first on all the sentences, then paragraphs, and then on the whole text, this way, it might be able to form meaning and contexts, maybe we go even 1 layer lower.
# maybe the text could be splitted in lines and smaller versions

def get_batch(split):
  # here we could add the grammer and the semantics of languages to make this better try adding more logic like sentence level reasoning and word making
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size, ))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1 : i + block_size + 1] for i in ix])
  x, y = x.to(device), y.to(device)

  return x, y

@torch.no_grad()
def estimate_loss(): # Output: e.g., "NVIDIA GeForce RTX 2060"
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
  def __init__(self, head_size, n_embd):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias = False)
    self.query = nn.Linear(n_embd, head_size, bias = False)
    self.value = nn.Linear(n_embd, head_size, bias = False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(Dropout)  # Adjust dropout rate as needed

  def forward(self, x):
      B, T, C = x.shape
      k = self.key(x)
      q = self.query(x)

      wei = q @ k.transpose(-2, -1) * C ** -0.5
      wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
      wei = F.softmax(wei, dim=-1)
      wei = self.dropout(wei)  # Apply dropout to attention weights
      wei.to(device)


      v = self.value(x)
      out = wei @ v
      return out
  

class MultiHeadAttention(nn.Module):
   
  def __init__(self, n_heads, head_size):
      super().__init__()
      self.heads = nn.ModuleList([Head(head_size, n_embd) for _ in range(n_heads)])
      self.proj = nn.Linear(n_embd, n_embd)
      self.dropout = nn.Dropout(Dropout)  # Adjust dropout rate as needed

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim = -1)
    return self.proj(out)

class FeedForward(nn.Module):
   
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, n_embd * 4),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(Dropout)
    )

  def forward(self, x):
     return self.net(x)

class Block(nn.Module):
   
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa_heads = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa_heads(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layers)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))
    x = tok_emb + pos_emb
    x = self.blocks(x)  # (B, T, n_embd)
    x = self.ln_f(x)  # (B, T, vocab_size)
    logits = self.lm_head(x) # (B, T, vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):

    for _ in range(max_new_tokens):

      idx_cond = idx[:, -block_size:]

      logits, loss = self(idx_cond)

      logits = logits[:, -1, :]

      probs = F.softmax(logits, dim = -1)

      idx_next = torch.multinomial(probs, num_samples=1)

      idx = torch.cat((idx, idx_next), dim=1)

    return idx


model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr = 5e-1)


for iter in range(max_iters):
   
    if iter % eval_interval == 0:
      losses = estimate_loss()
      print(f"step {iter} : train loss {losses['train']:.4f}, val loss = {losses['val']:.4f}")

    xb, yb = get_batch('train')
   

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device )
print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) 


if __name__ == "__main__":
   with open('model.pth', 'wb') as f:
       torch.save(m.state_dict(), f)