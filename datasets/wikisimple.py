from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from utils.processing import train_tokenizer, prepare_raw_wikidata, load_tokenizer
import pandas as pd
import numpy as np
import torch


class WikiSimpleDataset(Dataset):

    def __init__(self, data_root, split, prepare=False):
        if prepare:
            prepare_raw_wikidata(data_root)
        if split == "train":
            self.src = open(f"{data_root}/preprocessed/train.src", "r").readlines()
            self.trg = open(f"{data_root}/preprocessed/train.trg", "r").readlines()
        elif split == "valid":
            self.src = open(f"{data_root}/preprocessed/valid.src", "r").readlines()
            self.trg = open(f"{data_root}/preprocessed/valid.trg", "r").readlines()
        elif split == "test":
            self.src = open(f"{data_root}/preprocessed/valid.src", "r").readlines()
            self.trg = open(f"{data_root}/preprocessed/valid.trg", "r").readlines()

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        X = self.src[idx]
        y = self.trg[idx]

        return X, y


def collate_sentences(tokenizer: Tokenizer) -> (torch.LongTensor, torch.LongTensor):
    def collate_fn(batch):
        batch = [list(i) for i in zip(*batch)]
        X_batch = tokenizer.encode_batch(batch[0])
        y_batch = tokenizer.encode_batch(batch[1])
        X_batch = torch.LongTensor([x.ids for x in X_batch])
        y_batch = torch.LongTensor([y.ids for y in y_batch])
        return X_batch, y_batch
    return collate_fn


if __name__ == '__main__':
    dataset = WikiSimpleDataset("../data/WikiSimple-translated/", split="train", prepare=False)
    # tokenizer = train_tokenizer("../data/WikiSimple-translated/preprocessed", "../")
    tokenizer = load_tokenizer("../tokenizer-wikisimple.json")
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_sentences(tokenizer))
    X, y = next(iter(loader))
    print(X.shape, y.shape)
