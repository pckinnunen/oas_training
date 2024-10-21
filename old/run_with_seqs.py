import torch
from torch import nn
from torch.optim import Adam
from mlm import MLM
import pandas as pd

seq_df = pd.read_csv('heavy_seq_small.csv', index_col = 0)

# instantiate the language model

from x_transformers import TransformerWrapper, Encoder

transformer = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 512,
        depth = 6,
        heads = 8
    )
)

# plugin the language model into the MLM trainer

trainer = MLM(
    transformer,
    mask_token_id = 2,          # the token id reserved for masking
    pad_token_id = 0,           # the token id for padding
    mask_prob = 0.15,           # masking probability for masked language modeling
    replace_prob = 0.90,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
    mask_ignore_token_ids = []  # other tokens to exclude from masking, include the [cls] and [sep] here
) #.cuda()

# optimizer

opt = Adam(trainer.parameters(), lr=3e-4)

# one training step (do this for many steps in a for loop, getting new `data` each time)

data = torch.randint(0, 20000, (8, 1024)) #.cuda()
int_data = torch.randint(0, 20, (8, 1024))
ohe_data = torch.nn.functional.one_hot(int_data)
# t = 'abcdefghijk'
# data = torch.tensor(
#     [t[i] for i in torch.randint(0, len(t), 8*1024)]
# )


loss = trainer(ohe_data)
loss.backward()
opt.step()
opt.zero_grad()

# after much training, the model should have improved for downstream tasks
print('done')
torch.save(transformer, f'./pretrained-model.pt')