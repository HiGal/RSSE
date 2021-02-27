import pandas as pd
import numpy as np
from argparse import ArgumentParser
import os


def df_to_pairs(path, out_path, split='train', lib='huggingface'):
    tmp_df = pd.read_csv(path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if lib == 'fairseq':
        tmp_df_ru = tmp_df[['target_x', 'target_y']]
        tmp_df_en = tmp_df[['src', 'dst']]
        tmp_df_en.to_csv(f"{out_path}/" + split + '.en.txt', sep=str('\t'), index=False, header=False)
        tmp_df_ru.to_csv(f"{out_path}/" + split + '.ru.txt', sep=str('\t'), index=False, header=False)
    elif lib == 'huggingface':
        tmp_df_source = tmp_df['target_x']
        tmp_df_source = tmp_df_source.append(tmp_df['src'], ignore_index=True)
        tmp_df_source = tmp_df_source.reindex(np.random.permutation(tmp_df_source.index))
        tmp_df_source.to_csv(f"{out_path}/" + split + ".source")
        tmp_df_target = tmp_df['target_y']
        tmp_df_target = tmp_df_target.append(tmp_df['dst'], ignore_index=True)
        tmp_df_target = tmp_df_target.reindex(np.random.permutation(tmp_df_target.index))
        tmp_df_target.to_csv(f"{out_path}/" + split + ".target")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("path", help="Path to the raw data")
    parser.add_argument("out_path", help="Path where formatted data will store")
    parser.add_argument("--split", help="Optional [train,valid,test]. Default: train", default="train")
    parser.add_argument("--lib", help="Framework for which prepare data", default="huggingface")

    args = parser.parse_args()

    df_to_pairs(args.path, args.out_path, args.split, args.lib)
