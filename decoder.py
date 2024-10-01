import torch
from utils import MultiHeadAttention

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden_dim=1024, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, n_heads, d_model // n_heads, d_model // n_heads)
        self.mha2 = MultiHeadAttention(d_model, n_heads, d_model // n_heads, d_model // n_heads)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, ffn_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ffn_hidden_dim, d_model),
            torch.nn.Dropout(dropout)
        )
        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.layernorm3 = torch.nn.LayerNorm(d_model)

    def forward(self, input, encoder_output, src_mask, tgt_mask):
        att = self.mha1(input, padding_mask=tgt_mask, add_decoder_mask=True)
        att = self.layernorm1(input + att)
        att2 = self.mha2(att, encoder_output, src_mask)
        att2 = self.layernorm2(att + att2)
        ffn = self.ffn(att2)
        output = self.layernorm3(att2 + ffn)
        return output
    
class Decoder(torch.nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super(Decoder, self).__init__()
        self.layers = torch.nn.ModuleList([DecoderLayer(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, input, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            input = layer(input, encoder_output, src_mask, tgt_mask)
        return input