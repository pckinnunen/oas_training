import torch
from torch import nn
from torch.optim import Adam
from mlm import MLM

# instantiate the language model

from x_transformers import TransformerWrapper, Encoder
import pandas as pd
import transformers
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

N_SEQS = 1024
N_VAL_SEQS = 1024

def plot_aa_abundance(seq_df):

    seq_len = [len(s) for s in seq_df['sequence']]
    max_len = max(seq_len)
    seq_list = []
    for sl, s in zip(seq_len, seq_df['sequence'].values):
        if sl == max_len:
            seq_list.append(list(s))
        else:
            seq_list.append(list(s) + list('-' * (max_len - sl)))
    plot_df = pd.DataFrame(
        data = seq_list
    )
    fig, ax = plt.subplots()
    sns.histplot(
        data = plot_df.melt(), 
        y = 'variable',
        x = 'value',
        multiple = 'fill',
        stat = 'proportion',
        ax = ax
    )




seq_df = pd.read_csv('./heavy_seqs_small.csv', index_col = 0)
vocab_file = '/Users/pckinnunen/miniforge3/lib/python3.10/site-packages/antiberty/trained_models/vocab.txt'
seq_subset = seq_df.sample(N_SEQS + N_VAL_SEQS)

sequences_all = seq_subset.values.tolist()
sequences = sequences_all[:N_SEQS]
sequences_validate = sequences_all[N_SEQS+1:]
# sequences = sequences[:2048]
# Following: https://github.com/jeffreyruffolo/AntiBERTy/blob/fd8b42271d1710a16e7fb4c55da3f027d947d177/antiberty/AntiBERTyRunner.py#L39
plot_aa_abundance(seq_subset.iloc[:N_SEQS,:])
plot_aa_abundance(seq_subset.iloc[N_SEQS+1:, :])


tokenizer = transformers.BertTokenizer(
    vocab_file = 'vocab.txt',
    do_lower_case = False
    ) 

#From antiberty, but I'm masking later in the training process not here
# for s in sequences:
#     for i, c in enumerate(s):
#         if c == '_':
#             print('editing')
#             s[i] = '[MASK]'


sequences = [" ".join(s) for s in sequences]
sequences_validate = [" ".join(s) for s in sequences_validate]
tokenizer_out = tokenizer(
    sequences,
    return_tensors='pt',
    padding = True,
)

tokenizer_out_validate = tokenizer(
    sequences_validate,
    return_tensors='pt',
    padding = True,
)
tokens = tokenizer_out['input_ids'].to('mps')
attention_mask = tokenizer_out['attention_mask'].to('mps')

tokens_validate = tokenizer_out_validate['input_ids'].to('mps')
attention_mask_validate = tokenizer_out_validate['attention_mask'].to('mps')

assert 1 == 2
# tokens = tokenizer_out['input_ids']
# attention_mask = tokenizer_out['attention_mask']

transformer = TransformerWrapper(
    num_tokens = 25,
    max_seq_len = 256,
    attn_layers = Encoder(
        dim = 64,
        depth = 6,
        heads = 4
    ),
    emb_dim=2
).to('mps')

# plugin the language model into the MLM trainer
print('trainer')
trainer = MLM(
    transformer,
    mask_token_id = 2,          # the token id reserved for masking
    pad_token_id = 0,           # the token id for padding
    mask_prob = 0.15,           # masking probability for masked language modeling
    replace_prob = 0.90,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
    mask_ignore_token_ids = [0,1,2,3,4]  # other tokens to exclude from masking, include the [cls] and [sep] here
) #.cuda()

# optimizer

opt = Adam(trainer.parameters(), lr=3e-4)

# one training step (do this for many steps in a for loop, getting new `data` each time)

# data = torch.randint(0, 20000, (8, 1024)) #.cuda()
# int_data = torch.randint(0, 20, (8, 1024))
# ohe_data = torch.nn.functional.one_hot(int_data)
# t = 'abcdefghijk'
# data = torch.tensor(
#     [t[i] for i in torch.randint(0, len(t), 8*1024)]
# )

training_loader = torch.utils.data.DataLoader(tokens, batch_size=64, shuffle=True)
validate_loader = torch.utils.data.DataLoader(tokens_validate, batch_size=64, shuffle=True)

ll = []
vll = []
N_EPOCHS = 100
print(f'train for {N_EPOCHS} epochs')
assert 1 == 2
for epoch in range(N_EPOCHS):
    epoch_loss = 0
    for i_b, batch in enumerate(training_loader):
        transformer.training = True
        opt.zero_grad()

        loss = trainer(batch)

        # print(f'epoch: {epoch}\tloss: {loss:.3f}')
        # ll.append(loss)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    ll.append(epoch_loss)

    val_loss = 0
    transformer.training = False
    with torch.no_grad():
        for i, validate_data in enumerate(validate_loader):
            loss = trainer(validate_data)
            val_loss += loss.item()
    print(f'epoch: {epoch+1}\t loss: {epoch_loss}\tval loss: {val_loss}')
    vll.append(val_loss)

plt.figure()
# plt.plot(np.arange(N_EPOCHS), np.array([loss.cpu().detach().numpy() for loss in ll]), label = 'loss')
# plt.plot(np.arange(N_EPOCHS), np.array([loss.cpu().detach().numpy() for loss in vll]), label = 'Val loss')

plt.plot(np.arange(N_EPOCHS), np.array(ll), label = 'loss')
plt.plot(np.arange(N_EPOCHS), np.array(vll), label = 'Val loss')
plt.legend()
plt.show()
# print('train step')
# loss = trainer(tokens)
# loss.backward()
# opt.step()
# opt.zero_grad()

# after much training, the model should have improved for downstream tasks
# print('done')

torch.save(transformer, f'./pretrained-model.pt')