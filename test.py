import torch
import nltk
import argparse
from encoder import Encoder
from decoder import Decoder
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

from utils import clean_text, build_vocab, sentence_to_indices, create_data_loader

nltk.download('punkt')

class Transformer(torch.nn.Module):
    def __init__(self, d_model, n_heads, n_layers, eng_vocab, fr_vocab, ffn_hidden_dim=1024, dropout_rate=0.1, device='cpu'):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, n_heads, n_layers, ffn_hidden_dim, dropout_rate)
        self.decoder = Decoder(d_model, n_heads, n_layers, ffn_hidden_dim, dropout_rate)
        self.linear = torch.nn.Linear(d_model, len(fr_vocab.keys()))

        self.eng_embedding = torch.nn.Embedding(len(eng_vocab.keys()), d_model, padding_idx=eng_vocab['<PAD>'], dtype=torch.float32)
        self.fr_embedding = torch.nn.Embedding(len(fr_vocab.keys()), d_model, padding_idx=fr_vocab['<PAD>'], dtype=torch.float32)

        self.device = device
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

    def positional_encoding(self, seq_len, d_model):        
        pos = torch.arange(seq_len, device=self.device).unsqueeze(1)
        i = torch.arange(d_model, device=self.device).unsqueeze(0)
        angle = pos / 10000 ** (2 * (i // 2) / d_model)
        pe = torch.zeros(seq_len, d_model, device=self.device)
        pe[:, 0::2] = torch.sin(angle[:, 0::2])
        pe[:, 1::2] = torch.cos(angle[:, 1::2])
        return pe 

    def forward(self, src, tgt):
        src_mask = (src == eng_vocab['<PAD>']).float()
        tgt_mask = (tgt == fr_vocab['<PAD>']).float()
                
        src = self.eng_embedding(src)
        tgt = self.fr_embedding(tgt)

        # positional encoding
        src += self.positional_encoding(src.size(1), src.size(2))
        tgt += self.positional_encoding(tgt.size(1), tgt.size(2))

        encoder_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.linear(output)

        # reshape the output to batch_size x emb_dim x seq_len
        output = output.transpose(1, 2)
        
        return output

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    eng_train = open('ted-talks-corpus/train.en', 'r')
    fr_train = open('ted-talks-corpus/train.fr', 'r')
    eng_test = open('ted-talks-corpus/test.en', 'r')
    fr_test = open('ted-talks-corpus/test.fr', 'r')

    eng_train_lines = eng_train.readlines()
    fr_train_lines = fr_train.readlines()

    eng_test_lines = eng_test.readlines()
    fr_test_lines = fr_test.readlines()

    eng_train.close()
    fr_train.close()

    # Tokenize the English and French sentences
    eng_train_lines = [line.strip() for line in eng_train_lines]
    fr_train_lines = [line.strip() for line in fr_train_lines]
    eng_test_lines = [line.strip() for line in eng_test_lines]
    fr_test_lines = [line.strip() for line in fr_test_lines]

    tokenized_train_eng = [word_tokenize(clean_text(sentence)) for sentence in eng_train_lines]
    tokenized_train_fr = [word_tokenize(clean_text(sentence), language="french") for sentence in fr_train_lines]
    tokenized_test_eng = [word_tokenize(clean_text(sentence)) for sentence in eng_test_lines]
    tokenized_test_fr = [word_tokenize(clean_text(sentence), language="french") for sentence in fr_test_lines]

    # Build the vocabulary
    eng_vocab = build_vocab(tokenized_train_eng)
    fr_vocab = build_vocab(tokenized_train_fr)

    test_eng = [sentence_to_indices(sentence, eng_vocab) for sentence in tokenized_test_eng]
    test_fr = [sentence_to_indices(sentence, fr_vocab) for sentence in tokenized_test_fr]

    test_eng = [sentence for sentence in test_eng if 128 >= len(sentence) >= 5]
    test_fr = [sentence for sentence in test_fr if 128 >= len(sentence) >= 5]

    print("Number of testidation samples:", len(test_eng))

    test_loader = create_data_loader(test_eng, test_fr, eng_vocab, fr_vocab, batch_size=1)

    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model)

    idx2word = {idx: word for word, idx in fr_vocab.items()}

    # translate the test sentences until EOS token is generated, also calculate the BLEU score (start with <SOS> token)
    model.eval()
    bleu_scores = []
    for i, (eng, fr) in enumerate(test_loader):
        eng = eng.to(device)
        fr = fr.to(device)

        output = model(eng, fr)
        output = torch.argmax(output, dim=2)
        output = output.squeeze(0).tolist()

        fr = fr.squeeze(0).tolist()
        fr = fr[1:] # remove <SOS> token
        output = output[:output.index(fr_vocab['<EOS>'])] # remove tokens after <EOS> token

        output = [idx2word.get(idx, '<UNK>') for idx in output]
        fr = [idx2word.get(idx, '<UNK>') for idx in fr]

        bleu_score = sentence_bleu([fr], output)
        bleu_scores.append(bleu_score)

        print("Test Sample:", i + 1)
        print("English:", ' '.join([list(eng_vocab.keys())[idx] for idx in eng.squeeze(0).tolist()]))
        print("French:", ' '.join([list(fr_vocab.keys())[idx] for idx in fr]))
        print("Translated French:", ' '.join([list(fr_vocab.keys())[idx] for idx in output]))
        print("BLEU Score:", bleu_score)
        print()
