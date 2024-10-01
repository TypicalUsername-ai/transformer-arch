from tokenizer import WordPieceTokenizer
from attention import Attention
from torch import nn, tensor
import torch


tokenizer = WordPieceTokenizer()

with open("./t8.shakespeare.txt", "r") as shakespeare:
    tokenizer.train(shakespeare)

embedding_dim = 256
embedding = nn.Embedding(tokenizer.tokenizer.get_vocab_size(True), embedding_dim)


sentence = """LAURENS
The ten-dollar founding father without a father
got a lot farther by working a lot harder,
by being a lot smarter,
by being a self-starter,
by fourteen, they placed him in charge of a
trading charter.
JEFFERSON
And every day while slaves were being slaughtered and carted
away across the waves, he struggled and kept his guard up.
Inside, he was longing for something to be a part of,
the brother was ready to beg, steal, borrow or barter.
MADISON
Then a hurricane came, and devastation reigned,
our man saw his future drip, dripping down the drain,
put a pencil to his temple, connected it to his brain,
and he wrote his first refrain, a testament to his pain."""

embedded = embedding(tensor(tokenizer.tokenize_ids(sentence)))
tokens = tokenizer.tokenize(sentence)

print(f"embedded tensor: {embedded.shape}")

linear_query = nn.Linear(embedding_dim , embedding_dim // 2)
linear_key = nn.Linear(embedding_dim , embedding_dim // 2)
linear_value = nn.Linear(embedding_dim , embedding_dim // 2)


attention = Attention(embedding_dim // 2)

attn_out, attended_weights = attention(linear_query(embedded), linear_key(embedded), linear_value(embedded))

print(f"after attention outputt: {attn_out.shape}, weights: {attended_weights.shape}")
print(f"{tokens[10]} =>> {sorted(set(zip(tokens, attended_weights[10].tolist())), key=lambda k: k[1])[0:5]}")
