import torch
from torch import nn
from torch.optim import Adam
from mlm import MLM
from utils import plot_aa_abundance, get_seqs
# instantiate the language model

# from x_transformers import TransformerWrapper, Encoder
import pandas as pd
import transformers
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Imports from kaggle example
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader



N_SEQS = 1024
N_VAL_SEQS = 64
seqs_train, seqs_test, seqs_df = get_seqs('./heavy_seqs_small.csv', N_SEQS, N_VAL_SEQS)
seqs_df.reset_index(inplace = True)

## Define tokenizer
tokenizer = transformers.BertTokenizer(
    vocab_file = 'vocab.txt',
    do_lower_case = False
    ) 
seqs_train = [" ".join(s) for s in seqs_train]
seqs_test = [" ".join(s) for s in seqs_test]
tokenizer_out_train = tokenizer(
    seqs_train,
    return_tensors='pt',
    padding = True,
)

tokenizer_out_test = tokenizer(
    seqs_test,
    return_tensors='pt',
    padding = True,
)
tokens_train = tokenizer_out_train['input_ids'].to('mps')
attention_mask_train = tokenizer_out_train['attention_mask'].to('mps')

tokens_test = tokenizer_out_test['input_ids'].to('mps')
attention_mask_test = tokenizer_out_test['attention_mask'].to('mps')

## Set up model (from: https://www.kaggle.com/code/mojammel/masked-language-model-with-pytorch-transformer)
device = "mps" 

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

## Define dataloader
def data_collate_fn(seqs):
    inputs = tokenizer(seqs, padding=True, return_tensors='pt')
    return inputs

class MyDataset(Dataset):
    def __init__(self, src, tokenizer):
        self.src = src
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        return src


## Define training loop
def train(model, dataloader):
    val_loss_cumulative = []
    train_loss_cumulative = []
    model.train()
    epochs = 20
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=0.0001)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            optim.zero_grad()
            input = batch['input_ids'].clone()
            src_mask = model.generate_square_subsequent_mask(batch['input_ids'].size(1))
            rand_value = torch.rand(batch.input_ids.shape)
            # in the kaggle example where this comes from, '[PAD]' is 0, [CLS] is 101, [SEP] is 102
            # I probably want to remove these AND the UNK token, but I need to change the indices
            rand_mask = (rand_value < 0.15) * (input != 0) * (input != 1) * (input != 2) * (input != 3)
            mask_idx=(rand_mask.flatten() == True).nonzero().view(-1)
            input = input.flatten()
            input[mask_idx] = 4
            input = input.view(batch['input_ids'].size())

            out = model(input.to(device), src_mask.to(device))
            loss = criterion(out.view(-1, ntokens), batch['input_ids'].view(-1).to(device))
            total_loss += loss
            epoch_loss += loss
            loss.backward()
            optim.step()
        train_loss_cumulative.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            input_val = tokenizer_out_test['input_ids'].clone()
            src_mask_val = model.generate_square_subsequent_mask(tokenizer_out_test['input_ids'].size(1))
            rand_value = torch.rand(input_val.shape)

            rand_mask = (rand_value < 0.15) * (input_val != 0) * (input_val != 1) * (input_val != 2) * (input_val != 3)
            mask_idx=(rand_mask.flatten() == True).nonzero().view(-1)
            input_val = input_val.flatten()
            input_val[mask_idx] = 4
            input_val = input_val.view(tokenizer_out_test['input_ids'].size())

            out = model(input_val.to(device), src_mask_val.to(device))
            val_loss = criterion(out.view(-1, ntokens), tokenizer_out_test.input_ids.view(-1).to(device))
            val_loss_cumulative.append(val_loss)
        # if (epoch+1)%5==0 or epoch==0 or epoch+1==epochs:
        # assert 1 == 2
        print("Epoch: {} -> loss: {}, val_loss: {}".format(
            epoch+1,
              total_loss/(
                  len(dataloader)*epoch+1
                  ),
                  val_loss))
    return train_loss_cumulative, val_loss_cumulative

def predict(model, input):
    model.eval()
    src_mask = model.generate_square_subsequent_mask(input.size(1))
    out = model(input.to(device), src_mask.to(device))
    out = out.topk(1).indices.view(-1)
    return out

dataset = MyDataset(seqs_train, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=data_collate_fn)

ntokens = tokenizer.vocab_size # the size of vocabulary
emsize = 10 # embedding dimension
nhid = 16 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
# Training Model

print('starting training')
train_loss, val_loss = train(model, dataloader)
train_loss = [t.detach().cpu().numpy() for t in train_loss]
val_loss = [t.detach().cpu().numpy() for t in val_loss]
plt.plot(np.arange(20), train_loss)
plt.plot(np.arange(20)+1, val_loss)
print('done')
