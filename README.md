# Russian Simple Sentence Evaluation

### Clone repository

```shell
git clone --recurse-submodules https://github.com/HiGal/RSSE.git
```

### Install dependencies

```shell
pip install -r requirements.txt
```

### Run on remote server
```shell
nohup python transformers/examples/legacy/seq2seq/finetune_trainer.py\
  --data_dir data/prepared\
  --model_name_or_path 'google/mt5-small'\
  --tokenizer_name 'google/mt5-small'\
  --output_dir "rsse_model" --num_train_epochs 1 --do_train \ 
  --n_val 5000  --freeze_embeds --freeze_encoder &
```
it is needed to not kill the process after detach ssh sessin
**Configure tensorboard**

from your local machine, run
```shell
ssh -N -f -L localhost:16006:localhost:6006 <user@remote>
```

on the remote machine, run:

tensorboard --logdir <path> --port 6006

Then, navigate to (in this example) http://localhost:16006 on your local machine.

explanation of ssh command:

`-N` : no remote commands

`-f` : put ssh in the background

`-L` [machine1]: [portA] : [<machine2>] : [<portB>]

forward [machine1]:[portA] (local scope) to [machine2]:[portB] (remote scope)