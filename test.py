import torch
import nltk
import argparse
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

from utils import clean_text, build_vocab, sentence_to_indices, create_data_loader

nltk.download('punkt')

def test(model, test_loader, fr_vocab, device='cpu'):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=fr_vocab['<PAD>'])

    fr_vocab_inv = {idx: word for word, idx in fr_vocab.items()}
        
    model.eval()
    with torch.no_grad():
        total_loss = 0
        bleu_scores = []
        
        for _, (eng, fr) in enumerate(test_loader):
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

                print(f'Predicted: {" ".join(pred_sentence)}')
                print(f'True: {" ".join(true_sentence)}')
                print(f'BLEU Score: {sentence_bleu([true_sentence], pred_sentence)}')
                print()

        avg_bleu = sum(bleu_scores) / len(bleu_scores)

        print(f'Test Loss: {total_loss / len(test_loader)}, Avg BLEU Score: {avg_bleu}')

    return total_loss / len(test_loader), avg_bleu

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

    test(model, test_loader, fr_vocab, device=device)