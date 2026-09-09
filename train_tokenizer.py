# train_tokenizer.py
# ------------------
# STEP 2
#
# Neural networks dont understand words directly.
# They understand numbers.
#
# SentencePiece breaks Urdu text into small pieces (subwords)
# and gives each piece an ID number.
#
# Example idea:
#   "دریائے" maybe becomes pieces like ["دریا", "ئے"]
# Then each piece becomes a number like [452, 891]
#
# Assignment wants vocab size = 8000
# Also I keep <ans> and </ans> as special full tokens.

import os
import csv
import sentencepiece as spm
import config


def collect_text(tsv_paths, out_txt):
    """
    Make one big text file from train + valid.
    SentencePiece will learn common pieces from this text.
    """
    folder = os.path.dirname(out_txt)
    if folder != "":
        os.makedirs(folder, exist_ok=True)

    count = 0
    out = open(out_txt, "w", encoding="utf-8")

    for path in tsv_paths:
        f = open(path, encoding="utf-8")
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # I add both input sentence and question text
            out.write(row["source"].strip() + "\n")
            out.write(row["target"].strip() + "\n")
            count = count + 2
        f.close()

    out.close()
    print("wrote", count, "lines to", out_txt)
    return out_txt


def train_sentencepiece(train_txt):
    """Train the actual tokenizer model."""
    os.makedirs(config.TOKENIZER_DIR, exist_ok=True)

    # user_defined_symbols means:
    # please never split <ans> and </ans>
    # keep them as one special token each
    spm.SentencePieceTrainer.train(
        input=train_txt,
        model_prefix=config.SP_MODEL_PREFIX,
        vocab_size=config.VOCAB_SIZE,
        model_type="unigram",
        character_coverage=0.9995,  # high because Urdu has many characters
        pad_id=config.PAD_ID,
        unk_id=config.UNK_ID,
        bos_id=config.BOS_ID,
        eos_id=config.EOS_ID,
        user_defined_symbols=["<ans>", "</ans>"],
        num_threads=4,
    )
    print("tokenizer saved as", config.SP_MODEL_FILE)


def quick_demo():
    """Just print one example so I can see if tokenizer works."""
    sp = spm.SentencePieceProcessor(model_file=config.SP_MODEL_FILE)
    sample = "پاکستان کا سب سے بڑا دریا <ans> دریائے سندھ </ans> ہے۔"

    pieces = sp.encode(sample, out_type=str)
    ids = sp.encode(sample, out_type=int)

    print("sample pieces:", pieces)
    print("sample ids:", ids)
    print("decoded back:", sp.decode(ids))


def main():
    train_txt = os.path.join(config.DATA_DIR, "spm_train.txt")
    paths = [config.TRAIN_TSV, config.VALID_TSV]

    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError("Missing " + p + ". Run prepare_data.py first.")

    collect_text(paths, train_txt)
    train_sentencepiece(train_txt)
    quick_demo()
    print("Tokenizer training done :)")

    # save tokenizer copy to Google Drive (safe on Colab)
    try:
        from save_to_drive import save_important_files_to_drive
        save_important_files_to_drive()
    except Exception as e:
        print("Drive save skipped:", e)


if __name__ == "__main__":
    main()
