import torch
from utils import MultiHeadAttention

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, d_model // n_heads, d_model // n_heads)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, d_model),
            torch.nn.Dropout(0.1)
        )
        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)

    def forward(self, input):
        # print(input.shape)
        att = self.mha(input)
        # print(att.shape)
        att = self.layernorm1(input + att)
        ffn = self.ffn(att)
        output = self.layernorm2(att + ffn)
        return output
  
class Encoder(torch.nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super(Encoder, self).__init__()
        self.layers = torch.nn.ModuleList([EncoderLayer(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, input):
        # print(input.shape)
        for layer in self.layers:
            input = layer(input)
        return input