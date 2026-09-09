# prepare_data.py
# ---------------
# STEP 1 of my project
#
# What this file does (in simple words):
# 1) download UQA dataset from huggingface
# 2) remove questions that have no answer
# 3) find the sentence that contains the answer
# 4) put <ans> and </ans> tags around the answer
# 5) save train.tsv and valid.tsv
# 6) also prepare Wiki-UQA as out-of-domain test

import os
import re
import csv
from datasets import load_dataset
from tqdm import tqdm
import config


# Urdu and English full stop / question marks etc.
# I use this to split paragraph into sentences
sentence_end_pattern = re.compile(r"(?<=[۔.؟!?\n])\s+")


def find_answer_sentence(context, answer_start, answer_text):
    """
    The dataset gives a full paragraph (context).
    But for question generation, I only want the ONE sentence
    that has the answer inside it.

    Example:
      context = many sentences...
      answer = دریائے سندھ
      I find which sentence has that answer and return only that sentence.
    """

    # bad answer position
    if answer_start is None or answer_start < 0:
        return None

    # sometimes answer_start is wrong, so I search for the answer text myself
    looks_wrong = False
    if answer_start >= len(context):
        looks_wrong = True
    else:
        # check first few characters match
        check_len = min(5, len(answer_text))
        if not context[answer_start:].startswith(answer_text[:check_len]):
            looks_wrong = True

    if looks_wrong:
        found = context.find(answer_text)
        if found == -1:
            return None
        answer_start = found

    answer_end = answer_start + len(answer_text)

    # find where this sentence starts
    # I look at all sentence breaks before the answer and take the last one
    start = 0
    for match in sentence_end_pattern.finditer(context[:answer_start]):
        start = match.end()

    # find where this sentence ends
    end = len(context)
    match = sentence_end_pattern.search(context[answer_end:])
    if match:
        end = answer_end + match.start()

    sentence = context[start:end].strip()
    if sentence == "":
        return None

    # also return where answer starts inside THIS sentence
    local_start = answer_start - start
    return sentence, local_start, answer_text


def add_answer_tags(sentence, local_start, answer_text):
    """
    Put tags around the answer so the model knows what to ask about.

    Before:
      پاکستان کا سب سے بڑا دریا دریائے سندھ ہے۔

    After:
      پاکستان کا سب سے بڑا دریا <ans> دریائے سندھ </ans> ہے۔
    """

    end = local_start + len(answer_text)

    # make sure the answer text is really there
    if sentence[local_start:end] != answer_text:
        pos = sentence.find(answer_text)
        if pos == -1:
            return None
        local_start = pos
        end = pos + len(answer_text)

    # cut sentence into 3 parts and put tags in middle
    left_part = sentence[:local_start]
    right_part = sentence[end:]
    tagged = left_part + " <ans> " + answer_text + " </ans> " + right_part

    # remove extra spaces
    tagged = re.sub(r"\s+", " ", tagged).strip()
    return tagged


def process_split(dataset_split, max_examples=None):
    """
    Go through one split (train or validation) and make clean pairs:
      source = sentence with <ans> tags
      target = the real question
    """
    rows = []
    skipped = 0

    for ex in tqdm(dataset_split, desc="preparing examples"):
        # skip questions that cannot be answered
        if ex.get("is_impossible", False) == True:
            skipped += 1
            continue

        answer = (ex.get("answer") or "").strip()
        question = (ex.get("question") or "").strip()
        context = (ex.get("context") or "").strip()
        answer_start = ex.get("answer_start", -1)

        # empty stuff is useless
        if answer == "" or question == "" or context == "":
            skipped += 1
            continue

        found = find_answer_sentence(context, answer_start, answer)
        if found is None:
            skipped += 1
            continue

        sentence, local_start, answer_text = found
        source = add_answer_tags(sentence, local_start, answer_text)
        if source is None:
            skipped += 1
            continue

        # skip too long or too short examples
        if len(source) > config.MAX_SRC_CHARS or len(question) > config.MAX_TGT_CHARS:
            skipped += 1
            continue
        if len(source) < 10 or len(question) < 5:
            skipped += 1
            continue

        # save one training example
        rows.append({"source": source, "target": question})

        # if I only want a small sample for testing
        if max_examples is not None and len(rows) >= max_examples:
            break

    print("  kept", len(rows), "examples, skipped", skipped)
    return rows


def save_tsv(rows, path):
    """Save list of dicts into a .tsv file (tab separated)."""
    folder = os.path.dirname(path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)

    f = open(path, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=["source", "target"], delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    f.close()
    print("  wrote", path)


def main():
    print("Loading UQA dataset from Hugging Face...")
    print("This may take a little time the first time.")
    ds = load_dataset(config.HF_DATASET)

    print("\nPreparing train data...")
    train_rows = process_split(ds["train"], config.MAX_TRAIN_EXAMPLES)
    save_tsv(train_rows, config.TRAIN_TSV)

    print("\nPreparing validation data...")
    valid_rows = process_split(ds["validation"], config.MAX_VALID_EXAMPLES)
    save_tsv(valid_rows, config.VALID_TSV)

    print("\nLoading Wiki-UQA for out-of-domain test...")
    try:
        ood = load_dataset(config.HF_OOD_DATASET)
        # on huggingface this dataset only has "train" split
        # but I will use it as my OOD test set
        ood_rows = process_split(ood["train"])
        save_tsv(ood_rows, config.OOD_TSV)
    except Exception as e:
        print("  could not prepare Wiki-UQA:", e)
        print("  its ok, I can still train without OOD for now")

    print("\nData preparation finished :)")

    # save data copy to Google Drive (safe on Colab)
    try:
        from save_to_drive import save_important_files_to_drive
        save_important_files_to_drive()
    except Exception as e:
        print("Drive save skipped:", e)


if __name__ == "__main__":
    main()
