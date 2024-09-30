import torch
import numpy as np

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, query_dim, value_dim):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.query_dim = query_dim
        self.value_dim = value_dim
        
        self.q_w = torch.nn.Linear(d_model, n_heads * query_dim)
        self.k_w = torch.nn.Linear(d_model, n_heads * query_dim)
        self.v_w = torch.nn.Linear(d_model, n_heads * value_dim)

        self.linear = torch.nn.Linear(n_heads * value_dim, d_model)

        self.softmax = torch.nn.Softmax(dim=-1)

    # q, k, v: batch_size x n_heads x seq_len x dim
    def attention(self, q, k, v, mask=None):
        d_k = k.size(-1)
        scores = q @ k.transpose(-2, -1) / np.sqrt(d_k)
        if mask is not None:
            # TODO: Implement masking
            pass

        # multiply with value
        output = scores @ v

        # concatenate heads
        output = output.transpose(1, 2).contiguous().view(q.size(0), -1, self.n_heads * self.value_dim)

        # linear layer
        output = self.linear(output)

        return output
    
    # input: batch_size x seq_len x d_model
    def forward(self, input, encoder_output=None):
        batch_size = input.size(0)
        seq_len = input.size(1)

        q = self.q_w(input).view(batch_size, seq_len, self.n_heads, self.query_dim)
        if encoder_output is None:
            k = self.k_w(input).view(batch_size, seq_len, self.n_heads, self.query_dim)
            v = self.v_w(input).view(batch_size, seq_len, self.n_heads, self.value_dim)
        else:
            k = self.k_w(encoder_output).view(batch_size, seq_len, self.n_heads, self.query_dim)
            v = self.v_w(encoder_output).view(batch_size, seq_len, self.n_heads, self.value_dim)

        # Transpose to batch_size x n_heads x seq_len x dim
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = self.attention(q, k, v)

        return output