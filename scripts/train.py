import torch
from torch.utils.data import DataLoader

from custom_models.default_tansformer import Trainer, Transformer
from custom_models.transformer import make_model
from utils.processing import load_tokenizer
from dataset.wikisimple import WikiSimpleDataset, collate_sentences
import yaml
from addict import Dict
from fire import Fire
from transformers import AutoTokenizer

if __name__ == '__main__':
    with open("../configs/default_transformer.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    config = Fire(Dict(config))


    train_dataset = WikiSimpleDataset(config.datasets.root, split=config.datasets.split.train,
                                      prepare=config.datasets.prepare)
    val_dataset = WikiSimpleDataset(config.datasets.root, split=config.datasets.split.valid,
                                    prepare=config.datasets.prepare)

    params = config.parameters
    device = torch.device(params.device)

    # tokenizer = load_tokenizer(params.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
    train_iterator = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True,
                                collate_fn=collate_sentences(tokenizer))
    valid_iterator = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=True,
                                collate_fn=collate_sentences(tokenizer))
    tokenizer.model_max_length = params.max_len

    params.src_vocab_size = tokenizer.vocab_size
    params.trg_vocab_size = tokenizer.vocab_size
    # Model hyperparameters
    params.src_pad_idx = tokenizer.pad_token_id

    model = Transformer(
        device,
        params
    ).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_normal_(p)

    # model = make_model(tokenizer.get_vocab_size(), tokenizer.get_vocab_size(), params.num_encoder_layers,
    #                    params.embedding_size,h=params.num_heads, dropout=params.dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(params.optimizer.lr))

    if "ReduceLROnPlateau" in params.scheduler:
        sched_params = params.scheduler.ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=sched_params.factor, patience=sched_params.factor, verbose=sched_params.verbose
        )
    else:
        scheduler = None

    criterion = torch.nn.CrossEntropyLoss(ignore_index=params.src_pad_idx)

    trainer = Trainer(model, tokenizer, criterion, optimizer, scheduler, config.trainer)

    trainer.train(train_iterator, valid_iterator)
