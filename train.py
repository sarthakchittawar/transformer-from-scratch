import re
import torch
import nltk
from tqdm import tqdm
from encoder import Encoder
from decoder import Decoder
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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

def train(model, train_loader, val_loader, fr_vocab, num_epochs=10, lr=0.0001, device='cpu'):
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=fr_vocab['<PAD>'])

    for epoch in tqdm(range(num_epochs)):
        model.train()
        for i, (eng, fr) in enumerate(train_loader):
            eng, fr = eng.to(device), fr.to(device)
            optimizer.zero_grad()
            output = model(eng, fr)
            loss = criterion(output, fr)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch: {epoch+1}, Iteration: {i+1}, Loss: {loss.item()}')
        
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, (eng, fr) in enumerate(val_loader):
                eng, fr = eng.to(device), fr.to(device)
                output = model(eng, fr)
                loss = criterion(output, fr)
                total_loss += loss.item()
            print(f'Epoch: {epoch+1}, Validation Loss: {total_loss / len(val_loader)}')

    return model
    
# unit testing for Encoder & Decoder

eng_train = open('ted-talks-corpus/train.en', 'r')
fr_train = open('ted-talks-corpus/train.fr', 'r')
eng_val = open('ted-talks-corpus/dev.en', 'r')
fr_val = open('ted-talks-corpus/dev.fr', 'r')

eng_train_lines = eng_train.readlines()
fr_train_lines = fr_train.readlines()

eng_val_lines = eng_val.readlines()
fr_val_lines = fr_val.readlines()

eng_train.close()
fr_train.close()

# Tokenize the English and French sentences
eng_train_lines = [line.strip() for line in eng_train_lines]
fr_train_lines = [line.strip() for line in fr_train_lines]
eng_val_lines = [line.strip() for line in eng_val_lines]
fr_val_lines = [line.strip() for line in fr_val_lines]

tokenized_train_eng = [word_tokenize(clean_text(sentence)) for sentence in eng_train_lines]
tokenized_train_fr = [word_tokenize(clean_text(sentence)) for sentence in fr_train_lines]
tokenized_val_eng = [word_tokenize(clean_text(sentence)) for sentence in eng_val_lines]
tokenized_val_fr = [word_tokenize(clean_text(sentence)) for sentence in fr_val_lines]

print(tokenized_train_eng[87])
print(tokenized_train_fr[87])

# Build the vocabulary
eng_vocab = build_vocab(tokenized_train_eng)
fr_vocab = build_vocab(tokenized_train_fr)

# Convert the sentences to indices
train_eng = [sentence_to_indices(sentence, eng_vocab) for sentence in tokenized_train_eng]
train_fr = [sentence_to_indices(sentence, fr_vocab) for sentence in tokenized_train_fr]

val_eng = [sentence_to_indices(sentence, eng_vocab) for sentence in tokenized_val_eng]
val_fr = [sentence_to_indices(sentence, fr_vocab) for sentence in tokenized_val_fr]

# print("Ignoring sentences with length less than 5 or greater than 20...")
# tokenized_eng_sentences = [sentence for sentence in tokenized_eng if 20 >= len(sentence) >= 5]
# tokenized_fr_sentences = [sentence for sentence in tokenized_fr if 20 >= len(sentence) >= 5]
train_eng = [sentence for sentence in train_eng if 128 >= len(sentence) >= 5]
train_fr = [sentence for sentence in train_fr if 128 >= len(sentence) >= 5]
val_eng = [sentence for sentence in val_eng if 128 >= len(sentence) >= 5]
val_fr = [sentence for sentence in val_fr if 128 >= len(sentence) >= 5]

print("Number of training samples:", len(train_eng))
print("Number of validation samples:", len(val_eng))

# Create the data loaders
train_loader = create_data_loader(train_eng, train_fr, eng_vocab, fr_vocab, batch_size=64)
val_loader = create_data_loader(val_eng, val_fr, eng_vocab, fr_vocab, batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = Transformer(512, 8, 6, eng_vocab, fr_vocab, device)

# Train the model
model = train(model, train_loader, val_loader, fr_vocab, num_epochs=10, lr=0.0001, device=device)