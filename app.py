# app.py
# ------
# Simple website using Flask.
#
# User does this:
# 1) paste Urdu sentence
# 2) type the answer text
# 3) click Generate
#
# Website shows:
# - sentence with <ans> tags
# - greedy question
# - beam search question

import os
import torch
import sentencepiece as spm
from flask import Flask, render_template, request

import config
from model import Seq2Seq
from decode import generate_question
from train import get_device

app = Flask(__name__)

# these will be loaded once when needed
DEVICE = get_device()
MODEL = None
SP = None


def ensure_model_loaded():
    """
    Load model and tokenizer if they exist.
    Returns True if ready, False if not trained yet.
    """
    global MODEL, SP

    if MODEL is not None:
        return True

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")

    if not os.path.exists(ckpt_path):
        return False
    if not os.path.exists(config.SP_MODEL_FILE):
        return False

    print("Loading model for website...")
    SP = spm.SentencePieceProcessor(model_file=config.SP_MODEL_FILE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    MODEL = Seq2Seq(ckpt["vocab_size"]).to(DEVICE)
    MODEL.load_state_dict(ckpt["model_state"])
    MODEL.eval()
    print("Model loaded.")
    return True


def mark_answer(sentence, answer):
    """
    Put <ans> </ans> around the answer inside the sentence.

    If user already wrote the tags, I leave it as it is.
    """
    sentence = sentence.strip()
    answer = answer.strip()

    if "<ans>" in sentence and "</ans>" in sentence:
        return sentence

    if answer == "":
        return None

    pos = sentence.find(answer)
    if pos == -1:
        # answer text not found in sentence
        return None

    left = sentence[:pos]
    right = sentence[pos + len(answer):]
    tagged = left + " <ans> " + answer + " </ans> " + right
    tagged = tagged.replace("  ", " ").strip()
    return tagged


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    sentence = ""
    answer = ""

    if request.method == "POST":
        sentence = request.form.get("sentence", "")
        answer = request.form.get("answer", "")

        ready = ensure_model_loaded()
        if ready == False:
            error = "Model not trained yet. First run prepare_data.py, train_tokenizer.py and train.py"
        else:
            source = mark_answer(sentence, answer)
            if source is None:
                error = "I could not find the answer inside the sentence. Please check spelling."
            else:
                outs = generate_question(MODEL, SP, source, DEVICE)
                result = {
                    "source": source,
                    "greedy": outs["greedy"],
                    "beam": outs["beam"],
                }

    return render_template(
        "index.html",
        result=result,
        error=error,
        sentence=sentence,
        answer=answer,
    )


if __name__ == "__main__":
    # run local website
    print("Open this in browser: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
