import torch
from utils import MultiHeadAttention

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden_dim=1024, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, d_model // n_heads, d_model // n_heads)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, ffn_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ffn_hidden_dim, d_model),
            torch.nn.Dropout(dropout)
        )
        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)

    def forward(self, input, input_mask):
        # print(input.shape)
        att = self.mha(input, padding_mask=input_mask)
        # print(att.shape)
        att = self.layernorm1(input + att)
        ffn = self.ffn(att)
        output = self.layernorm2(att + ffn)
        return output
  
class Encoder(torch.nn.Module):
    def __init__(self, d_model, n_heads, n_layers, ffn_hidden_dim=1024, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = torch.nn.ModuleList([EncoderLayer(d_model, n_heads, ffn_hidden_dim, dropout) for _ in range(n_layers)])

    def forward(self, input, input_mask):
        # print(input.shape)
        for layer in self.layers:
            input = layer(input, input_mask)
        return input