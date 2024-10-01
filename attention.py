import torch
import torch.nn.functional as F
from torch import nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

if __name__ == '__main__':
    # Example usage:
    hidden_size = 25
    print(f"hiddent_size is {hidden_size} (this comes from embedding in real life)")
    attention = Attention(hidden_size)

    query = torch.randn(10, hidden_size)  # Example query tensor
    key = torch.randn(10, hidden_size)    # Example key tensor
    value = torch.randn(10, hidden_size)  # Example value tensor

    print(f"query[0]: {query[0]}")
    output, attention_weights = attention(query, key, value)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attention_weights.shape)
    print(f"outptu[0]: {output[0]}")
    print(f"weights[0]: {attention_weights[0]}")
