import torch
import yaml
from addict import Dict
from fire import Fire
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm

from custom_models.default_tansformer import Transformer
from utils.processing import load_tokenizer, shorten_sentence

if __name__ == '__main__':

    with open("../configs/default_transformer.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    config = Fire(Dict(config))

    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")

    params = config.parameters
    device = torch.device(params.device)
    tokenizer.model_max_length = params.max_len

    params.src_vocab_size = tokenizer.vocab_size
    params.trg_vocab_size = tokenizer.vocab_size
    # Model hyperparameters
    params.src_pad_idx = tokenizer.pad_token_id
    model = Transformer(
        device,
        params
    ).to(device)
    model.load_state_dict(
        torch.load("models/default_transformer/my_checkpoint.pth.tar", map_location='cuda:0')['state_dict'])
    sentences = []
    with open("../data/pairs/public_test_only.csv") as f:
        reader = f.readlines()
        i = 0
        for row in tqdm(reader):
            sentence = row
            translated_sentence = shorten_sentence(
                model, sentence, tokenizer, device, max_length=50
            )

            sentences.append('"'+translated_sentence+'"\n')
    open("answer.txt", "w").writelines(sentences)
