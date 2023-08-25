import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import spacy
import random
import sys

from torchtext.data.metrics import bleu_score
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from tqdm import tqdm

spacy_eng = spacy.load('en_core_web_sm')
spacy_ger = spacy.load('de_core_news_sm')
print('Load Successfully')


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenize_eng, lower=True, init_token='<sos>', eos_token='<eos>')

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german, english),
                                                    root='./Translation Data')

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


def translate_sentence(model, sentence, german, english, device, max_length=50):
    spacy_ger = spacy.load('de_core_news_sm')
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <sos> and <eos>
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    sentense_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(sentense_tensor)

    outputs = [english.vocab.stoi['<sos>']]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        if output.argmax(1).item() == english.vocab.stoi['<eos>']:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    return translated_sentence[1:]


def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)['src']
        trg = vars(example)['trg']

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size)
        hidden, cell = self.encoder(source)

        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess  # 生成的可能是错误的 所以两种情况都要保留

        return outputs


num_epochs = 100
learning_rate = 0.0005
batch_size = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_size_encoder = len(german.vocab)
print('German vocab size:', input_size_encoder)
input_size_decoder = len(english.vocab)
print('English vocab size:', input_size_decoder)
output_size = len(english.vocab)
encoder_embedding_size = 256
decoder_embedding_size = 256
hidden_size = 1024
num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_key=lambda x: len(x.src),
    sort_within_batch=True,
    device=device
)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, encoder_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers,
                      decoder_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# sentence = 'ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen.'

for epoch in range(num_epochs):
    model.train()
    for batch_idx, batch in enumerate(tqdm(train_iterator)):
        input_data = batch.src.to(device)
        target = batch.trg.to(device)
        # print(batch)
        print(input_data[:3])
        # print(target)
        output = model(input_data, target)
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        step += 1

    score = bleu(test_data[1:10], model, german, english, device)
    print(f'Bleu score {score * 100:.2f}')
