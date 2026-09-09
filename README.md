# Gen AI Project 1 – Urdu Question Generation

This is my Gen AI course project.

**Idea:** take an Urdu sentence, mark the answer with `<ans> ... </ans>`, and the model generates a question for that answer.

Example:

- Input: `پاکستان کا سب سے بڑا دریا <ans> دریائے سندھ </ans> ہے۔`
- Output: something like `پاکستان کا سب سے بڑا دریا کون سا ہے؟`

I trained everything from scratch (no pretrained transformers / no ready-made seq2seq model).

---

## What I built

Pipeline:

1. Prepare UQA data
2. Train SentencePiece tokenizer (vocab size 8000)
3. Train encoder-decoder model with attention
4. Generate questions (greedy + beam search)
5. Evaluate with BLEU / ROUGE-L / perplexity / unk rate
6. Human evaluation sheet + Cohen’s kappa
7. Simple Flask website for demo

### Model (assignment requirements)

- Encoder: 2-layer BiLSTM
- Decoder: 2-layer LSTM
- Attention: Bahdanau
- Training: teacher forcing + cross entropy + Adam
- Decoding: greedy and beam search (beam size 4)

---

## Project files

| File | What it does |
|------|----------------|
| `config.py` | all settings in one place |
| `prepare_data.py` | download/clean UQA, add `<ans>` tags |
| `train_tokenizer.py` | SentencePiece tokenizer |
| `dataset.py` | loads tsv into batches |
| `model.py` | encoder + attention + decoder |
| `train.py` | training + resume/checkpoint |
| `decode.py` | greedy + beam |
| `evaluate.py` | automatic metrics |
| `human_eval.py` | human scoring + kappa |
| `save_to_drive.py` | backup to Google Drive (for Colab) |
| `app.py` | web demo |
| `predict.py` | terminal demo |
| `templates/index.html` | website page |

Note: `data/`, `tokenizer/`, and `checkpoints/` are not in git (too big / generated files). I kept them on Google Drive after Colab training.

---

## How to run

Install:

```bash
pip install -r requirements.txt
```

Then in order:

```bash
python prepare_data.py
python train_tokenizer.py
python train.py
python evaluate.py
python evaluate.py --beam
python evaluate.py --tsv data/ood_test.tsv
python app.py
```

Website: http://127.0.0.1:5000

Quick test without browser:

```bash
python predict.py --sentence "پاکستان کا سب سے بڑا دریا دریائے سندھ ہے۔" --answer "دریائے سندھ"
```

### Human eval

```bash
python human_eval.py sheet
python human_eval.py kappa data/reviewer_a.csv data/reviewer_b.csv
```

---

## Colab notes

I trained this on Google Colab with GPU because my laptop is slow for full training.

- Mount Drive first
- `save_to_drive.py` copies `data`, `tokenizer`, `checkpoints` to Drive
- `train.py` can resume from `last_checkpoint.pt` if Colab disconnects

If training is too heavy for a first test, in `config.py` you can temporarily set something like:

```python
MAX_TRAIN_EXAMPLES = 8000
NUM_EPOCHS = 5
```

For final run I used the full setup.

---

## Datasets / references

- UQA dataset: https://huggingface.co/datasets/uqa/UQA
- Out-of-domain Wiki-UQA: https://huggingface.co/datasets/uqa/Wiki-UQA
- UQA paper/code: https://github.com/sameearif/UQA

---

## Short viva line

We take an Urdu sentence with a marked answer, tokenize it with our own SentencePiece vocab, pass it through a from-scratch bidirectional LSTM encoder, use Bahdanau attention with a 2-layer LSTM decoder, and generate the Urdu question using greedy or beam search.
