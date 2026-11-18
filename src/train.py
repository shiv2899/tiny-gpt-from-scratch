import torch
from gpt import GPTLanguageModel, batch_size, block_size, max_iters, eval_interval, learning_rate

torch.manual_seed(1337)
with open('../data/input.txt','r',encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split=='train' else val_data
    ix = torch.randint(len(d)-block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x,y

model = GPTLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for it in range(max_iters):
    xb,yb = get_batch('train')
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward(); optimizer.step()
    if it % eval_interval == 0:
        print(f"step {it}: loss {loss.item():.4f}")
torch.save(model.state_dict(), '../checkpoints/gpt_model.pt')
print("Saved!")
