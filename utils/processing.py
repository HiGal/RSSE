import pandas as pd
import torch
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os

# spacy_ru = Russian()
#
#
# def tokenize(text):
#     return [tok.text for tok in spacy_ru.tokenizer(text)]


def prepare_raw_wikidata(data_path):
    files = {
        "train": "wiki_train_cleaned_translated_sd.csv",
        "valid": "wiki_dev_cleaned_translated_sd.csv",
        "test": "wiki_test_cleaned_translated_sd.csv"
    }
    file_mapping = {}
    for stage in files.keys():
        src = []
        target = []
        df = pd.read_csv(f"{data_path}/{files[stage]}", index_col=0)
        for idx, row in df.iterrows():
            src.append(row['target_x'] + "\n")
            target.append(row["target_y"] + "\n")

        file_mapping[stage + ".src"] = src
        file_mapping[stage + ".trg"] = target

    if not os.path.exists(f"{data_path}/preprocessed"):
        os.mkdir(f"{data_path}/preprocessed")

    for k, v in file_mapping.items():
        with open(f'{data_path}/preprocessed/{k}', 'w') as fp:
            fp.writelines(v)


def shorten_sentence(model, sentence, tokenizer, device, max_length=50):

    tokens = tokenizer.encode(sentence)

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(tokens.ids).unsqueeze(1).to(device)

    outputs = [tokenizer.token_to_id("[SOS]")]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)
        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == tokenizer.token_to_id("[EOS]"):
            break
    translated_sentence = tokenizer.decode(outputs)
    # remove start token
    return translated_sentence


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def train_tokenizer(raw_data_path, out_path):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    files = os.listdir(raw_data_path)
    files = [f"{raw_data_path}/{file_name}" for file_name in files if os.path.isfile(f"{raw_data_path}/{file_name}")]
    trainer = BpeTrainer(special_tokens=["[UNK]", "[SOS]", "[SEP]", "[EOS]", "[PAD]"])
    tokenizer.train(files=files, trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        pair="[SOS] $A [SEP] $B:1 [EOS]",
        special_tokens=[
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]"))
        ]
    )
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    tokenizer.save(f"{out_path}/tokenizer-wikisimple.json")
    return tokenizer


def load_tokenizer(path_to_tokenizer):
    if not os.path.exists(path_to_tokenizer):
        return train_tokenizer("../data/WikiSimple-translated/preprocessed", "../data/WikiSimple-translated/")
    return Tokenizer.from_file(path_to_tokenizer)

