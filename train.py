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
    def __init__(self, d_model, n_heads, n_layers, eng_vocab, fr_vocab):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, n_heads, n_layers)
        self.decoder = Decoder(d_model, n_heads, n_layers)
        self.linear = torch.nn.Linear(d_model, len(fr_vocab.keys()))
        self.softmax = torch.nn.Softmax(dim=-1)

        self.eng_embedding = torch.nn.Embedding(len(eng_vocab.keys()), d_model, padding_idx=eng_vocab['<PAD>'], dtype=torch.float32)
        self.fr_embedding = torch.nn.Embedding(len(fr_vocab.keys()), d_model, padding_idx=fr_vocab['<PAD>'], dtype=torch.float32)

    def forward(self, src, tgt):
        # TODO: Implement padding mask
        
        src = self.eng_embedding(src)
        tgt = self.fr_embedding(tgt)

        encoder_output = self.encoder(src)
        output = self.decoder(tgt, encoder_output)
        output = self.linear(output)
        output = self.softmax(output)

        # reshape the output to batch_size x emb_dim x seq_len
        output = output.transpose(1, 2)
        
        return output
        
    
def clean_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove the punctuation
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

    eng_padded = []
    fr_padded = []

    for eng_seq, fr_seq in zip(eng_sequences, fr_sequences):
        eng_padded.append(eng_seq + [eng_vocab['<PAD>']] * (512 - len(eng_seq)))
        fr_padded.append(fr_seq + [fr_vocab['<PAD>']] * (512 - len(fr_seq)))

    eng_tensor = torch.tensor(eng_padded)
    fr_tensor = torch.tensor(fr_padded)

    dataset = TensorDataset(eng_tensor, fr_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

def train(model, train_loader, val_loader, fr_vocab, num_epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=fr_vocab['<PAD>'])

    for epoch in tqdm(range(num_epochs)):
        model.train()
        for i, (eng, fr) in enumerate(train_loader):
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
                output = model(eng, fr)
                loss = criterion(output, fr)
                total_loss += loss.item()
            print(f'Epoch: {epoch+1}, Validation Loss: {total_loss / len(val_loader)}')

    return model
    
# unit testing for Encoder & Decoder

eng = open('ted-talks-corpus/train.en', 'r')
fr = open('ted-talks-corpus/train.fr', 'r')

eng_lines = eng.readlines()
fr_lines = fr.readlines()

eng.close()
fr.close()

# Tokenize the English and French sentences
eng_lines = [line.strip() for line in eng_lines]
fr_lines = [line.strip() for line in fr_lines]

tokenized_eng = [word_tokenize(clean_text(sentence)) for sentence in eng_lines]
tokenized_fr = [word_tokenize(clean_text(sentence)) for sentence in fr_lines]

print(tokenized_eng[87])
print(tokenized_fr[87])

# Build the vocabulary
eng_vocab = build_vocab(tokenized_eng)
fr_vocab = build_vocab(tokenized_fr)

# Convert the sentences to indices
eng_sentences = [sentence_to_indices(sentence, eng_vocab) for sentence in tokenized_eng]
fr_sentences = [sentence_to_indices(sentence, fr_vocab) for sentence in tokenized_fr]

# print("Ignoring sentences with length less than 5 or greater than 20...")
# tokenized_eng_sentences = [sentence for sentence in tokenized_eng if 20 >= len(sentence) >= 5]
# tokenized_fr_sentences = [sentence for sentence in tokenized_fr if 20 >= len(sentence) >= 5]

# Split the data into training and validation sets
train_eng, val_eng, train_fr, val_fr = train_test_split(eng_sentences, fr_sentences, test_size=0.2)

print("Number of training samples:", len(train_eng))
print("Number of validation samples:", len(val_eng))

# Create the data loaders
train_loader = create_data_loader(train_eng, train_fr, eng_vocab, fr_vocab)
val_loader = create_data_loader(val_eng, val_fr, eng_vocab, fr_vocab)

# Initialize the model
model = Transformer(512, 8, 6, eng_vocab, fr_vocab)

# Train the model
model = train(model, train_loader, val_loader, fr_vocab, num_epochs=10, lr=0.001)