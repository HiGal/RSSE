import os

import torch
from torch import nn, optim
from tqdm import tqdm
from einops import rearrange

from utils.processing import shorten_sentence, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(
            self,
            device,
            kwargs
    ):
        super(Transformer, self).__init__()
        self.d_model = kwargs.embedding_size
        self.src_word_embedding = nn.Embedding(kwargs.src_vocab_size, kwargs.embedding_size)
        # self.pos_enc = PositionalEncoding(kwargs.embedding_size, kwargs.dropout, kwargs.max_len)
        self.src_position_embedding = nn.Embedding(kwargs.max_len, kwargs.embedding_size)
        self.trg_word_embedding = nn.Embedding(kwargs.trg_vocab_size, kwargs.embedding_size)
        # self.trg_pos_enc = PositionalEncoding(kwargs.embedding_size, kwargs.dropout, kwargs.max_len)
        self.trg_position_embedding = nn.Embedding(kwargs.max_len, kwargs.embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            kwargs.embedding_size,
            kwargs.num_heads,
            kwargs.num_encoder_layers,
            kwargs.num_decoder_layers,
            kwargs.forward_expansion,
            kwargs.dropout,
            kwargs.activation
        )
        self.fc_out = nn.Linear(kwargs.embedding_size, kwargs.trg_vocab_size)
        self.dropout = nn.Dropout(kwargs.dropout)
        self.src_pad_idx = kwargs.src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
                .unsqueeze(1)
                .expand(src_seq_length, N)
                .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
                .unsqueeze(1)
                .expand(trg_seq_length, N)
                .to(self.device)
        )
        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )
        # embed_src = self.pos_enc(self.src_word_embedding(src)*math.sqrt(self.d_model))
        # embed_trg = self.pos_enc(self.trg_word_embedding(trg)*math.sqrt(self.d_model))

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        # out = out.permute(1,0,2)
        out = self.fc_out(out)
        return out


class Trainer:

    def __init__(self,
                 model,
                 tokenizer,
                 criterion,
                 optimizer,
                 scheduler=None,
                 cfg=None
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer

        self.load_model = False
        self.save_model = True

        self.writer = SummaryWriter()

        # Training hyperparameters
        self.num_epochs = cfg.epochs
        self.cfg = cfg

        self.step = 0

        self.model = model

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.criterion = criterion

        if not os.path.exists(self.cfg.model_dir):
            os.makedirs(self.cfg.model_dir)

        if self.cfg.load_checkpoint:
            load_checkpoint(torch.load(f"{cfg.model_dir}/my_checkpoint.pth.tar"), self.model, self.optimizer)

    def train_one_epoch(self, train_iterator):
        self.model.train()
        train_losses = []

        for batch_idx, (input, target) in tqdm(enumerate(train_iterator), total=len(train_iterator)):
            # Get input and targets and get to cuda
            # inp_data = batch.target_x.to(self.device)
            # target = batch.target_y.to(self.device)
            inp_data = input.permute(1, 0).to(self.device)
            target = target.permute(1, 0).to(self.device)

            # Forward prop
            output = self.model(inp_data, target[:-1])

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            self.optimizer.zero_grad()

            loss = self.criterion(output, target)
            train_losses.append(loss.item())

            # Back prop
            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

            # Gradient descent step
            self.optimizer.step()

            # plot to tensorboard
            if (self.step + 1) % self.cfg.verbose_step == 0:
                self.model.eval()
                sentence = self.cfg.valid_sentence
                translated_sentence = shorten_sentence(
                    self.model, sentence, self.tokenizer, self.device, max_length=self.cfg.max_length
                )

                print(f"\nTranslated example sentence: \n {translated_sentence}")
                self.model.train()
            self.writer.add_scalar("Training loss", loss.item(), global_step=self.step)
            self.step += 1

        mean_loss = sum(train_losses) / len(train_losses)
        self.scheduler.step(mean_loss)
        return mean_loss

    def validation(self, val_iterator):
        val_losses = []
        self.model.eval()
        for batch_idx, (input, target) in tqdm(enumerate(val_iterator), total=len(val_iterator)):
            # Get input and targets and get to cuda
            inp_data = input.permute(1, 0).to(self.device)
            target = target.permute(1, 0).to(self.device)

            # Forward prop
            output = self.model(inp_data, target[:-1])

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            loss = self.criterion(output, target)
            val_losses.append(loss.item())

            # plot to tensorboard
            self.writer.add_scalar("Validation loss", loss.item(), global_step=self.step)
        mean_loss = sum(val_losses) / len(val_losses)
        return mean_loss

    def train(self, train_iterator, val_iterator):
        sentence = self.cfg.valid_sentence
        for epoch in range(self.num_epochs):
            print(f"[Epoch {epoch} / {self.num_epochs}]")

            if self.save_model:
                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, f"{self.cfg.model_dir}/my_checkpoint.pth.tar")

            self.model.eval()
            translated_sentence = shorten_sentence(
                self.model, sentence, self.tokenizer, self.device, max_length=self.cfg.max_length
            )

            print(f"\nTranslated example sentence: \n {translated_sentence}")

            train_loss = self.train_one_epoch(train_iterator)
            val_loss = self.validation(val_iterator)

            print(f"Train loss: {train_loss} Validation Loss: {val_loss}")
