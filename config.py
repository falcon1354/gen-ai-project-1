# config.py
# ---------
# I put all important settings in this one file.
# So if I want to change epochs or batch size, I only change it here.
# Other files will import these values.

# folders where I save stuff
DATA_DIR = "data"
TOKENIZER_DIR = "tokenizer"
CHECKPOINT_DIR = "checkpoints"

# prepared data files (tab separated)
TRAIN_TSV = "data/train.tsv"
VALID_TSV = "data/valid.tsv"
OOD_TSV = "data/ood_test.tsv"  # Wiki-UQA out of domain test

# SentencePiece tokenizer settings
# assignment says vocab size should be 8000
VOCAB_SIZE = 8000
SP_MODEL_PREFIX = "tokenizer/urdu_sp"
SP_MODEL_FILE = "tokenizer/urdu_sp.model"

# special token ids
# pad = filler so batches have same length
# unk = unknown word
# bos = beginning of sentence
# eos = end of sentence
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# if sentence/question is too long, I skip it during data prep
MAX_SRC_CHARS = 400
MAX_TGT_CHARS = 150

# max tokens after tokenizer
MAX_SRC_LEN = 80
MAX_TGT_LEN = 40

# model size
# not too big because I will train on Colab
EMBED_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 2   # assignment wants 2 layer encoder/decoder
DROPOUT = 0.3    # randomly turns off some neurons so it doesnt overfit too much

# training settings
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 12
TEACHER_FORCING_RATIO = 0.5  # half the time give correct next word
CLIP_GRAD = 1.0              # stop gradients from becoming crazy big

# when generating questions
BEAM_SIZE = 4       # assignment says beam 3 to 5
MAX_DECODE_LEN = 40

# for first test on Colab you can set these to small numbers like 5000
# if None, it uses all data
MAX_TRAIN_EXAMPLES = None
MAX_VALID_EXAMPLES = None

# huggingface dataset names
HF_DATASET = "uqa/UQA"
HF_OOD_DATASET = "uqa/Wiki-UQA"

# Google Drive saving (for Colab)
# If True, code will also copy results to Drive so they stay safe
SAVE_TO_DRIVE = True

# This folder will be created in your Google Drive
# You can change the name if you want
DRIVE_RESULTS_DIR = "/content/drive/MyDrive/GenAI_Project1_results"
