# human_eval.py
# -------------
# This file is for HUMAN evaluation (not automatic scores).
#
# Assignment asks:
# - 2 people score 50 outputs
# - score fluency, relevance, answerability
# - then calculate Cohen's kappa (how much both people agree)
#
# How I use this file:
#
# 1) Make scoring sheet:
#    python human_eval.py sheet
#
# 2) After both reviewers fill their CSV files:
#    python human_eval.py kappa data/reviewer_a.csv data/reviewer_b.csv

import os
import csv
import sys
import random

import sentencepiece as spm
from sklearn.metrics import cohen_kappa_score

import config
from dataset import QGDataset
from decode import generate_question
from evaluate import load_model
from train import get_device


def make_sheet():
    # Make a CSV file with 50 model outputs.
    # Then humans will open it in Excel/Google Sheets and give scores.

    print("Making human evaluation sheet...")

    # settings
    tsv_path = config.VALID_TSV
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    out_csv = "data/human_eval_sheet.csv"
    n = 50
    seed = 42

    # allow simple optional args:
    # python human_eval.py sheet 30
    # python human_eval.py sheet 30 my_sheet.csv
    if len(sys.argv) >= 3:
        n = int(sys.argv[2])
    if len(sys.argv) >= 4:
        out_csv = sys.argv[3]

    # check files
    if not os.path.exists(tsv_path):
        print("ERROR: validation data not found:", tsv_path)
        return
    if not os.path.exists(ckpt_path):
        print("ERROR: model not found. Train first using train.py")
        return

    device = get_device()
    print("Using device:", device)

    # load tokenizer and model
    sp = spm.SentencePieceProcessor(model_file=config.SP_MODEL_FILE)
    model, _ = load_model(ckpt_path, device)

    # load dataset
    ds = QGDataset(tsv_path, config.SP_MODEL_FILE)
    print("Total examples available:", len(ds.pairs))

    # make list of indexes: 0,1,2,3,...
    idxs = []
    i = 0
    while i < len(ds.pairs):
        idxs.append(i)
        i = i + 1

    # shuffle so we get random 50 examples
    random.Random(seed).shuffle(idxs)

    # take only first n
    selected = []
    i = 0
    while i < n and i < len(idxs):
        selected.append(idxs[i])
        i = i + 1

    # generate questions and store in rows
    rows = []
    i = 0
    while i < len(selected):
        idx = selected[i]
        src_text = ds.pairs[idx][0]
        tgt_text = ds.pairs[idx][1]

        # model makes greedy + beam questions
        outs = generate_question(model, sp, src_text, device)

        row = {}
        row["id"] = i + 1
        row["source"] = src_text
        row["reference_question"] = tgt_text
        row["greedy_output"] = outs["greedy"]
        row["beam_output"] = outs["beam"]

        # these empty columns are for humans to fill later
        row["fluency"] = ""
        row["relevance"] = ""
        row["answerability"] = ""
        row["notes"] = ""

        rows.append(row)

        print("Done", i + 1, "/", len(selected))
        i = i + 1

    # save csv file
    folder = os.path.dirname(out_csv)
    if folder != "":
        os.makedirs(folder, exist_ok=True)

    # utf-8-sig helps Excel show Urdu correctly (adds BOM)
    f = open(out_csv, "w", encoding="utf-8-sig", newline="")
    fieldnames = [
        "id",
        "source",
        "reference_question",
        "greedy_output",
        "beam_output",
        "fluency",
        "relevance",
        "answerability",
        "notes",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        writer.writerow(row)

    f.close()

    print("")
    print("Saved sheet to:", out_csv)
    print("")
    print("If Excel shows weird text, open the CSV in Google Sheets instead,")
    print("or in Excel use: Data -> From Text/CSV -> File Origin = UTF-8")
    print("")
    print("Next steps for me/my team:")
    print("1) Make 2 copies of this file")
    print("   - reviewer_a.csv")
    print("   - reviewer_b.csv")
    print("2) Each person scores alone (do not copy each other)")
    print("3) Give score from 1 to 5 for:")
    print("   - fluency (does Urdu sound natural?)")
    print("   - relevance (is question about the sentence?)")
    print("   - answerability (can marked answer answer this question?)")
    print("4) Then run kappa command")


def read_one_column(path, column_name):
    # Read one score column from a filled csv file
    # Example column_name: fluency

    scores = []
    # utf-8-sig also reads normal utf-8 files fine
    f = open(path, encoding="utf-8-sig")
    reader = csv.DictReader(f)

    for row in reader:
        value = row[column_name].strip()

        if value == "":
            print("ERROR: missing score in file:", path)
            print("Column:", column_name)
            f.close()
            return None

        # convert text number to int
        scores.append(int(float(value)))

    f.close()
    return scores


def compute_kappa():
    # Compare reviewer A and reviewer B scores.
    # Cohen's kappa tells agreement.
    # 1.0 = perfect agreement
    # 0.0 = agreement by chance only

    if len(sys.argv) < 4:
        print("Usage:")
        print("python human_eval.py kappa data/reviewer_a.csv data/reviewer_b.csv")
        return

    file_a = sys.argv[2]
    file_b = sys.argv[3]

    if not os.path.exists(file_a):
        print("ERROR: file not found:", file_a)
        return
    if not os.path.exists(file_b):
        print("ERROR: file not found:", file_b)
        return

    print("Reviewer A file:", file_a)
    print("Reviewer B file:", file_b)
    print("")

    columns = ["fluency", "relevance", "answerability"]

    for col in columns:
        scores_a = read_one_column(file_a, col)
        scores_b = read_one_column(file_b, col)

        if scores_a is None or scores_b is None:
            return

        if len(scores_a) != len(scores_b):
            print("ERROR: both files should have same number of rows")
            return

        kappa = cohen_kappa_score(scores_a, scores_b)
        print("Cohen's kappa for", col, "=", round(kappa, 3))


def main():
    # very simple menu using command line words
    if len(sys.argv) < 2:
        print("How to use this file:")
        print("")
        print("Make scoring sheet:")
        print("  python human_eval.py sheet")
        print("")
        print("Calculate agreement:")
        print("  python human_eval.py kappa data/reviewer_a.csv data/reviewer_b.csv")
        return

    command = sys.argv[1]

    if command == "sheet":
        make_sheet()
    elif command == "kappa":
        compute_kappa()
    else:
        print("Unknown command:", command)
        print("Use: sheet   or   kappa")


if __name__ == "__main__":
    main()
