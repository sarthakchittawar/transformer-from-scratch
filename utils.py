import re
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def build_vocab(sentences):
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)
    
    vocab['<UNK>'] = len(vocab)
    vocab['<SOS>'] = len(vocab)
    vocab['<EOS>'] = len(vocab)
    vocab['<PAD>'] = len(vocab)
    
    return vocab

def sentence_to_indices(sentence, vocab):
    sentence2 = ['<SOS>'] + sentence + ['<EOS>']
    return [vocab.get(word, vocab['<UNK>']) for word in sentence2]

def create_data_loader(eng_sequences, fr_sequences, eng_vocab, fr_vocab, batch_size=32):    
    max_len_eng = max([len(sentence) for sentence in eng_sequences])
    max_len_fr = max([len(sentence) for sentence in fr_sequences])

    print("Max length of English sentences:", max_len_eng)
    print("Max length of French sentences:", max_len_fr)

    eng_padded = []
    fr_padded = []

    for eng_seq, fr_seq in zip(eng_sequences, fr_sequences):
        eng_padded.append(eng_seq + [eng_vocab['<PAD>']] * (128 - len(eng_seq)))
        fr_padded.append(fr_seq + [fr_vocab['<PAD>']] * (128 - len(fr_seq)))

    eng_tensor = torch.tensor(eng_padded)
    fr_tensor = torch.tensor(fr_padded)

    dataset = TensorDataset(eng_tensor, fr_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

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
        scores = q @ k.transpose(-2, -1)

        if mask is not None:
            # scores: batch_size x n_heads x seq_len x seq_len
            # mask: batch_size x seq_len x seq_len
            mask = mask.unsqueeze(1)
            scores += mask

        scores = self.softmax(scores / np.sqrt(d_k))

        # multiply with value
        output = scores @ v

        # concatenate heads
        output = output.transpose(1, 2).contiguous().view(q.size(0), -1, self.n_heads * self.value_dim)

        # linear layer
        output = self.linear(output)

        return output
    
    # input: batch_size x seq_len x d_model
    def forward(self, input, encoder_output=None, padding_mask=None, add_decoder_mask=False):
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

        # padding mask shape: batch_size x seq_len
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).repeat(1, seq_len, 1)
            mask *= float('-inf')
            mask = torch.nan_to_num(mask, nan=0.0, neginf=float('-inf'))
        else:
            mask = None

        if add_decoder_mask:
            decoder_mask = torch.triu(torch.ones(batch_size, seq_len, seq_len), diagonal=1).to(input.device)

            mask += decoder_mask

        output = self.attention(q, k, v, mask)

        return output