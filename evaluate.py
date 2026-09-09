# evaluate.py
# -----------
# This file checks how good my trained model is.
#
# The assignment asks for these automatic scores:
# 1) BLEU-4
# 2) ROUGE-L
# 3) Perplexity
# 4) <unk> rate
#
# How I run it:
#   python evaluate.py
#   python evaluate.py --beam
#   python evaluate.py --tsv data/ood_test.tsv

import os
import math
import sys

import torch
import torch.nn as nn
from tqdm import tqdm
import sentencepiece as spm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import config
from dataset import make_loader
from model import Seq2Seq
from decode import greedy_decode, beam_search_decode, ids_to_text
from train import get_device


def lcs_length(list_a, list_b):
    # Longest Common Subsequence
    # I need this for ROUGE-L.
    # Simple meaning: how many words match in order (not always side by side).

    len_a = len(list_a)
    len_b = len(list_b)

    # make a table filled with zeros
    table = []
    for i in range(len_a + 1):
        row = []
        for j in range(len_b + 1):
            row.append(0)
        table.append(row)

    # fill the table
    i = 1
    while i <= len_a:
        j = 1
        while j <= len_b:
            if list_a[i - 1] == list_b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                up = table[i - 1][j]
                left = table[i][j - 1]
                if up > left:
                    table[i][j] = up
                else:
                    table[i][j] = left
            j = j + 1
        i = i + 1

    return table[len_a][len_b]


def rouge_l_score(pred_words, real_words):
    # ROUGE-L for one pair of questions
    # Higher score means more overlap with the real question.

    if len(pred_words) == 0:
        return 0.0
    if len(real_words) == 0:
        return 0.0

    lcs = lcs_length(pred_words, real_words)

    precision = lcs / len(pred_words)
    recall = lcs / len(real_words)

    if precision + recall == 0:
        return 0.0

    # this is the normal ROUGE-L formula
    beta = 1.2
    beta2 = beta * beta
    top = (1 + beta2) * precision * recall
    bottom = recall + (beta2 * precision)
    score = top / bottom
    return score


def load_model(checkpoint_path, device):
    # Load the saved model file
    print("Loading model from:", checkpoint_path)
    saved = torch.load(checkpoint_path, map_location=device)

    model = Seq2Seq(saved["vocab_size"])
    model = model.to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()  # eval mode = no training updates

    return model, saved


def compute_perplexity(model, loader, device):
    # Perplexity tells how "surprised" the model is.
    # Lower number is usually better.

    loss_fn = nn.CrossEntropyLoss(ignore_index=config.PAD_ID, reduction="sum")

    total_loss = 0.0
    total_tokens = 0

    # no_grad means we are only testing, not training
    with torch.no_grad():
        for batch in tqdm(loader, desc="perplexity"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_lens = batch["src_lens"].to(device)

            # teacher forcing off during testing
            outputs = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)

            # skip first token <s>
            pred = outputs[:, 1:]
            real = tgt[:, 1:]

            loss = loss_fn(
                pred.reshape(-1, pred.size(-1)),
                real.reshape(-1),
            )

            # count only real tokens, not PAD
            num_real = (real != config.PAD_ID).sum().item()
            total_loss = total_loss + loss.item()
            total_tokens = total_tokens + num_real

    if total_tokens == 0:
        return 0.0

    avg_loss = total_loss / total_tokens

    # protect from math overflow
    if avg_loss > 20:
        avg_loss = 20

    ppl = math.exp(avg_loss)
    return ppl


def evaluate_generation(model, sp, loader, device, use_beam):
    # Make questions with the model and compare with real questions.

    all_real = []      # real questions
    all_pred = []      # model questions
    unk_count = 0
    total_tok = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="generate"):
            src = batch["src"].to(device)
            src_lens = batch["src_lens"].to(device)

            # choose greedy or beam
            if use_beam == True:
                pred_batch = []
                batch_size = src.size(0)
                i = 0
                while i < batch_size:
                    one_src = src[i:i + 1]
                    one_len = src_lens[i:i + 1]
                    one_pred = beam_search_decode(model, one_src, one_len)
                    pred_batch.append(one_pred)
                    i = i + 1
            else:
                pred_batch = greedy_decode(model, src, src_lens)

            # compare each prediction with real question
            i = 0
            while i < len(pred_batch):
                pred_ids = pred_batch[i]
                pred_text = ids_to_text(sp, pred_ids)
                real_text = batch["tgt_text"][i]

                pred_words = pred_text.split()
                real_words = real_text.split()

                all_pred.append(pred_words)
                all_real.append([real_words])  # bleu wants list of references

                # count unknown tokens
                for token_id in pred_ids:
                    if token_id == config.UNK_ID:
                        unk_count = unk_count + 1

                if len(pred_ids) == 0:
                    total_tok = total_tok + 1
                else:
                    total_tok = total_tok + len(pred_ids)

                i = i + 1

    # ---- BLEU-4 ----
    # BLEU checks n-gram overlap (1 word, 2 words, 3 words, 4 words)
    smoothie = SmoothingFunction().method1
    bleu = corpus_bleu(
        all_real,
        all_pred,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    )
    bleu = bleu * 100

    # ---- ROUGE-L ----
    rouge_total = 0.0
    i = 0
    while i < len(all_pred):
        one_score = rouge_l_score(all_pred[i], all_real[i][0])
        rouge_total = rouge_total + one_score
        i = i + 1

    if len(all_pred) == 0:
        rouge = 0.0
    else:
        rouge = (rouge_total / len(all_pred)) * 100

    # ---- unk rate ----
    if total_tok == 0:
        unk_rate = 0.0
    else:
        unk_rate = (unk_count / total_tok) * 100

    # keep first 5 examples to show
    samples = []
    i = 0
    while i < len(all_pred) and i < 5:
        samples.append((all_real[i][0], all_pred[i]))
        i = i + 1

    result = {
        "bleu4": bleu,
        "rouge_l": rouge,
        "unk_rate": unk_rate,
        "num_examples": len(all_pred),
        "samples": samples,
    }
    return result


def main():
    # default values
    tsv_path = config.VALID_TSV
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    use_beam = False

    # very simple command line reading
    # examples:
    #   python evaluate.py
    #   python evaluate.py --beam
    #   python evaluate.py --tsv data/ood_test.tsv
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--beam":
            use_beam = True
        elif sys.argv[i] == "--tsv":
            i = i + 1
            tsv_path = sys.argv[i]
        elif sys.argv[i] == "--checkpoint":
            i = i + 1
            ckpt_path = sys.argv[i]
        i = i + 1

    # check files exist
    if not os.path.exists(ckpt_path):
        print("ERROR: model file not found.")
        print("Train the model first using train.py")
        return

    if not os.path.exists(tsv_path):
        print("ERROR: data file not found:", tsv_path)
        return

    device = get_device()
    print("Using device:", device)

    # load tokenizer
    sp = spm.SentencePieceProcessor(model_file=config.SP_MODEL_FILE)

    # load model
    model, saved = load_model(ckpt_path, device)
    print("Loaded epoch:", saved.get("epoch"))
    print("Saved val loss:", saved.get("val_loss"))

    # make data loader
    loader = make_loader(tsv_path, shuffle=False, batch_size=16)

    print("")
    print("Calculating perplexity...")
    ppl = compute_perplexity(model, loader, device)

    print("")
    print("Generating questions and calculating BLEU / ROUGE / UNK...")
    # make loader again so we start from beginning
    loader = make_loader(tsv_path, shuffle=False, batch_size=16)
    gen = evaluate_generation(model, sp, loader, device, use_beam)

    print("")
    print("========== RESULTS ==========")
    print("examples     :", gen["num_examples"])
    if use_beam == True:
        print("decoding     : beam")
    else:
        print("decoding     : greedy")
    print("BLEU-4       :", round(gen["bleu4"], 2))
    print("ROUGE-L      :", round(gen["rouge_l"], 2))
    print("Perplexity   :", round(ppl, 2))
    print("<unk> rate % :", round(gen["unk_rate"], 2))

    print("")
    print("Some examples:")
    for pair in gen["samples"]:
        real_q = pair[0]
        pred_q = pair[1]
        print("  REF:", " ".join(real_q))
        print("  HYP:", " ".join(pred_q))
        print("  ---")


if __name__ == "__main__":
    # nltk sometimes needs this download
    try:
        import nltk
        nltk.download("punkt", quiet=True)
    except Exception:
        pass

    main()
