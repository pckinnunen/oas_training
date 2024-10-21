from datasets import load_dataset

from transformers import AutoTokenizer

eli5 = load_dataset('eli5_category', split = 'train[:5000]')
eli5 = eli5.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained('distilbert/distilroberta-base')
eli5 = eli5.flatten()

def preprocess_function(ex):
    return tokenizer([" ".join(x) for x in ex['answers.text']])

tokenized_eli5 = eli5.map(
    preprocess_function,
    batched = True,
    num_proc = 4,
    remove_columns=eli5['train'].column_names,
)
