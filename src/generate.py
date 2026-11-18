import torch
from gpt import GPTLanguageModel
with open('../data/input.txt','r',encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
itos = {i:ch for i,ch in enumerate(chars)}
decode = lambda l: ''.join([itos[i] for i in l])
model = GPTLanguageModel(vocab_size)
model.load_state_dict(torch.load('../checkpoints/gpt_model.pt'))
model.eval()
context = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
