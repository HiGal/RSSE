from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import pandas as pd
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer


def prepare_data(files_dir, src_len=350, dst_len=300):
    # train_src = open(f"{files_dir}/train.src", "r").readlines() + open(f"{files_dir}/paraphraser.src").readlines()
    # train_trg = open(f"{files_dir}/train.trg", "r").readlines() + open(f"{files_dir}/paraphraser.trg").readlines()
    train_src = open(f"{files_dir}/train.src", "r").readlines()
    train_trg = open(f"{files_dir}/train.trg", "r").readlines()
    train_dict = {
        "input_text": train_src,
        "target_text": train_trg
    }
    val_src = open(f"{files_dir}/valid.src", "r").readlines()
    val_trg = open(f"{files_dir}/valid.trg", "r").readlines()
    val_dict = {
        "input_text": val_src,
        "target_text": val_trg
    }
    train_df = pd.DataFrame.from_dict(train_dict)
    train_df = train_df[train_df['input_text'].str.len() < src_len]
    train_df = train_df[train_df['target_text'].str.len() < dst_len]
    train_df = train_df.drop_duplicates()
    val_df = pd.DataFrame.from_dict(val_dict)
    return train_df, val_df


if __name__ == '__main__':
    model1_args = Seq2SeqArgs()
    model1_args.max_length = 128
    model1_args.max_seq_length = 128
    model1_args.num_beams = 4
    model1_args.no_cache = True
    model2_args = Seq2SeqArgs()
    model2_args.no_cache = True
    model2_args.max_length = 128
    model2_args.max_seq_length = 128
    model2_args.num_beams = 4
    # model1_args.num_train_epochs = 3
    # model_args.fp16 = True
    # model_args.use_multiprocessing = False
    # model_args.dataloader_num_workers = 80
    # model_args.train_batch_size = 4
    # model_args.max_length = 128
    # # model_args.acc
    # # model_args.src_lang = "ru_RU"
    # # model_args.tgt_lang = "ru_RU"
    # model_args.overwrite_output_dir = True
    # print(model_args)

    # train_data, eval_data = prepare_data("../data/WikiSimple-translated/preprocessed")
    #
    # print(train_data.shape)
    # print(eval_data.shape)

    print(model2_args)

    tokenizera = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    modela = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ru-en").cuda()
    tokenizerb = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    modelb = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ru").cuda()

    # modela = Seq2SeqModel(
    #     encoder_decoder_type="marian",
    #     encoder_decoder_name="Helsinki-NLP/opus-mt-ru-en",
    #     max_length=128
    # )
    # modelb = Seq2SeqModel(
    #     encoder_decoder_type="marian",
    #     encoder_decoder_name="Helsinki-NLP/opus-mt-en-ru",
    #     max_length=128
    # )

    # model.train_model(train_data, "./logs", eval_data=eval_data)
    batch_size = 64
    results = []
    with open("../data/pairs/public_test_only.csv", "r") as f:
        data = f.readlines()
        batches = len(data) // batch_size
        for i in tqdm(range(batches)):
            if i != batches - 1:
                input_batch = data[i * batch_size:(i + 1) * batch_size]
            else:
                input_batch = data[i * batch_size:]

            tokens = tokenizera.prepare_seq2seq_batch(src_texts=input_batch, return_tensors='pt')
            tokens = {k: v.cuda() for k, v in tokens.items()}
            intermediate_res = modela.generate(**tokens)
            intermediate_res = [tokenizera.decode(t, skip_special_tokens=True) for t in intermediate_res]
            tokens = tokenizerb.prepare_seq2seq_batch(src_texts=intermediate_res, return_tensors='pt')
            tokens = {k: v.cuda() for k, v in tokens.items()}
            res = modelb.generate(**tokens)
            res = ['"'+tokenizerb.decode(t, skip_special_tokens=True)+'"\n' for t in res]
            results += res

    open("answers.txt", "w").writelines(results)

    # to_predict = [
    #     "22 марта 1951 года боевая машина БМ-24 постановлением Совета Министров СССР №875-441сс была принята на вооружение Советской армии."
    # ]
    # intermediate = modela.predict(to_predict)
    # print(intermediate)
    # paraphrased = modelb.predict(intermediate)
    # print(paraphrased)
