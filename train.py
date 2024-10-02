import torch
import nltk
import wandb
import argparse
from tqdm import tqdm
from encoder import Encoder
from decoder import Decoder
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

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

def train(model, train_loader, val_loader, fr_vocab, num_epochs=10, lr=0.0001, device='cpu'):
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=fr_vocab['<PAD>'])

    fr_vocab_inv = {idx: word for word, idx in fr_vocab.items()}

    rouge = Rouge()

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

        train_loss = loss.item()
        
        model.eval()
        with torch.no_grad():
            total_loss = 0
            bleu_scores = []
            rouge_scores = []
            for i, (eng, fr) in enumerate(val_loader):
                eng, fr = eng.to(device), fr.to(device)
                output = model(eng, fr)
                loss = criterion(output, fr)
                total_loss += loss.item()

                # Calculate BLEU and ROUGE scores
                output_indices = output.argmax(dim=1).cpu().numpy()
                fr_indices = fr.cpu().numpy()
                for j in range(len(output_indices)):
                    pred_sentence = [fr_vocab_inv.get(idx, '<UNK>') for idx in output_indices[j] if idx != fr_vocab['<PAD>']]
                    true_sentence = [fr_vocab_inv.get(idx, '<UNK>') for idx in fr_indices[j] if idx != fr_vocab['<PAD>']]
                    bleu_scores.append(sentence_bleu([true_sentence], pred_sentence))
                    rouge_scores.append(rouge.get_scores(' '.join(pred_sentence), ' '.join(true_sentence))[0])

            avg_bleu = sum(bleu_scores) / len(bleu_scores)
            avg_rouge = {
                'rouge-1': sum([score['rouge-1']['f'] for score in rouge_scores]) / len(rouge_scores),
                'rouge-2': sum([score['rouge-2']['f'] for score in rouge_scores]) / len(rouge_scores),
                'rouge-l': sum([score['rouge-l']['f'] for score in rouge_scores]) / len(rouge_scores)
            }

            print(f'Epoch: {epoch+1}, Validation Loss: {total_loss / len(val_loader)}, BLEU Score: {avg_bleu}, ROUGE Scores: {avg_rouge}')

            if wandb.run:  
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': total_loss / len(val_loader),
                    'bleu_score': avg_bleu,
                    'rouge-1': avg_rouge['rouge-1'],
                    'rouge-2': avg_rouge['rouge-2'],
                    'rouge-l': avg_rouge['rouge-l'],
                    'epoch': epoch
                })
    

    return model, avg_bleu, avg_rouge, train_loss, total_loss / len(val_loader)
    
# unit testing for Encoder & Decoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--index', type=int, default=0)

    args = parser.parse_args()

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
    tokenized_train_fr = [word_tokenize(clean_text(sentence), language="french") for sentence in fr_train_lines]
    tokenized_val_eng = [word_tokenize(clean_text(sentence)) for sentence in eng_val_lines]
    tokenized_val_fr = [word_tokenize(clean_text(sentence), language="french") for sentence in fr_val_lines]

    # Build the vocabulary
    eng_vocab = build_vocab(tokenized_train_eng)
    fr_vocab = build_vocab(tokenized_train_fr)

    # Convert the sentences to indices
    train_eng = [sentence_to_indices(sentence, eng_vocab) for sentence in tokenized_train_eng]
    train_fr = [sentence_to_indices(sentence, fr_vocab) for sentence in tokenized_train_fr]

    val_eng = [sentence_to_indices(sentence, eng_vocab) for sentence in tokenized_val_eng]
    val_fr = [sentence_to_indices(sentence, fr_vocab) for sentence in tokenized_val_fr]

    train_eng = [sentence for sentence in train_eng if 128 >= len(sentence) >= 5]
    train_fr = [sentence for sentence in train_fr if 128 >= len(sentence) >= 5]
    val_eng = [sentence for sentence in val_eng if 128 >= len(sentence) >= 5]
    val_fr = [sentence for sentence in val_fr if 128 >= len(sentence) >= 5]

    print("Number of training samples:", len(train_eng))
    print("Number of validation samples:", len(val_eng))

    # Create the data loaders
    train_loader = create_data_loader(train_eng, train_fr, eng_vocab, fr_vocab, batch_size=64)
    val_loader = create_data_loader(val_eng, val_fr, eng_vocab, fr_vocab, batch_size=64)

    # Train the model
    device = torch.device(args.device)
    model = Transformer(args.d_model, args.n_heads, args.n_layers, eng_vocab, fr_vocab, dropout_rate=args.dropout_rate, device=device)

    wandb.init(project='transformer', entity='sarthakchittawar', config={'layers': args.n_layers, 'heads': args.n_heads, 'd_model': args.d_model, 'dropout_rate': args.dropout_rate})
    model, avg_bleu, avg_rouge, train_loss, val_loss = train(model, train_loader, val_loader, fr_vocab, num_epochs=10, lr=0.0001, device=device)
    print("BLEU Score:", avg_bleu)
    print("ROUGE Scores:", avg_rouge)

    torch.save(model, f'transformer_{args.index}.pth')