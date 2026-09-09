# predict.py
# ----------
# Small command line demo (no website needed).
#
# Example:
# python predict.py --sentence "پاکستان کا سب سے بڑا دریا دریائے سندھ ہے۔" --answer "دریائے سندھ"

import argparse
import os
import sentencepiece as spm

import config
from app import mark_answer
from decode import generate_question
from evaluate import load_model
from train import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pt"),
    )
    args = parser.parse_args()

    # add <ans> tags
    source = mark_answer(args.sentence, args.answer)
    if source is None:
        print("Could not find the answer in the sentence.")
        return

    # load model
    device = get_device()
    sp = spm.SentencePieceProcessor(model_file=config.SP_MODEL_FILE)
    model, _ = load_model(args.checkpoint, device)

    # generate
    outs = generate_question(model, sp, source, device)

    print("input :", source)
    print("greedy:", outs["greedy"])
    print("beam  :", outs["beam"])


if __name__ == "__main__":
    main()
