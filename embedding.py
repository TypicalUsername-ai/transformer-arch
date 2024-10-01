import torch
import torch.nn as nn

# Example vocabulary and corresponding embeddings
vocab = {"dog": 0, "cat": 1, "bird": 2}
embedding_dim = 2

# Embedding layer
embedding_layer = nn.Embedding(len(vocab), embedding_dim)

# Set specific embeddings for each word (fictional values for illustration)
embedding_layer.weight.data = torch.FloatTensor([[0.5, 1.2], [2.0, -0.8], [-1.3, 0.9]])

# Example: Embedding for the word "cat"
word_index = vocab["cat"]
cat_embedding = embedding_layer(torch.LongTensor([word_index]))

print("Vocabulary:", vocab)
print("Embedding dimension:", embedding_dim)
print("Embedding for 'cat':", cat_embedding.tolist())