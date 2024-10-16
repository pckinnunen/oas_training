import pandas as pd
import os

N_LIGHT = 1e6
N_HEAVY = 1e6

oas_files = os.listdir('./oas_data/')
heavy_files = [f for f in oas_files if 'Heavy_Bulk' in f]
light_files = [f for f in oas_files if 'Light_Bulk' in f]

print(f'Heavy:\n{heavy_files[:3]}\nLen = {len(heavy_files)}')
print(f'Light:\n{light_files[:3]}\nLen = {len(light_files)}')

light_seqs = []

i = 0
while len(light_seqs)<=N_LIGHT and i<=len(light_files):
    print(i)
    df = pd.read_csv(
        f'./oas_data/{light_files[i]}',
        sep=',',
        skiprows = 1
    )
    light_seqs.extend(df['sequence'].to_list())

    i+=1
print(len(light_seqs))

light_df = pd.DataFrame.from_dict(
    {
        'sequence': light_seqs
    }
)
light_df.to_csv('./light_seqs.csv')

heavy_seqs = []

i = 0
while len(heavy_seqs)<=N_HEAVY and i<=len(heavy_files):
    print(i)
    df = pd.read_csv(
        f'./oas_data/{heavy_files[i]}',
        sep=',',
        skiprows = 1
    )
    heavy_seqs.extend(df['sequence'].to_list())

    i+=1
print(len(heavy_seqs))

heavy_df = pd.DataFrame.from_dict(
    {
        'sequence': heavy_seqs
    }
)
heavy_df.to_csv('./heavy_seqs.csv')