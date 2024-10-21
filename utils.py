import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def get_seqs(seq_file, n_train_seqs, n_test_seqs):
    seq_df = pd.read_csv(seq_file, index_col = 0)
    seqs_subset_df = seq_df.sample(n_train_seqs + n_test_seqs)

    seqs_subset_list = seqs_subset_df['sequence'].values.tolist()
    seqs_train = seqs_subset_list[:n_train_seqs]
    seqs_test = seqs_subset_list[n_train_seqs:]
    return seqs_train, seqs_test, seqs_subset_df

