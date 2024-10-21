import pandas as pd

import transformers


seq_df = pd.read_csv('./heavy_seqs_small.csv', index_col = 0)
vocab_file = '/Users/pckinnunen/miniforge3/lib/python3.10/site-packages/antiberty/trained_models/vocab.txt'

sequences = seq_df['sequence'].values.tolist()

# Following: https://github.com/jeffreyruffolo/AntiBERTy/blob/fd8b42271d1710a16e7fb4c55da3f027d947d177/antiberty/AntiBERTyRunner.py#L39
tokenizer = transformers.BertTokenizer(
    vocab_file = 'vocab.txt',
    do_lower_case = False
    ) 

for s in sequences:
    for i, c in enumerate(s):
        if c == '_':
            s[i] = '[MASK]'

sequences = [" ".join(s) for s in sequences]
tokenizer_out = tokenizer(
    sequences,
    return_tensors='pt',
    padding = True,
)

tokens = tokenizer_out['input_ids'].to('mps')
attention_mask = tokenizer_out['attention_mask'].to('mps')




