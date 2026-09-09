# dataset.py
# ----------
# This file reads my TSV files and converts text into numbers
# so PyTorch can train on them.
#
# One example looks like:
#   source text -> list of token ids
#   target question -> list of token ids
#
# Then DataLoader groups many examples into a batch.

import csv
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import config


class QGDataset(Dataset):
    """
    QG = Question Generation dataset
    It just stores pairs of (sentence_with_ans_tags, question)
    """

    def __init__(self, tsv_path, sp_model_path, max_src=None, max_tgt=None):
        # load tokenizer
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)

        # max lengths
        if max_src is None:
            self.max_src = config.MAX_SRC_LEN
        else:
            self.max_src = max_src

        if max_tgt is None:
            self.max_tgt = config.MAX_TGT_LEN
        else:
            self.max_tgt = max_tgt

        # read all rows into a normal python list
        self.pairs = []
        f = open(tsv_path, encoding="utf-8")
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            src = row["source"].strip()
            tgt = row["target"].strip()
            if src != "" and tgt != "":
                self.pairs.append((src, tgt))
        f.close()

    def __len__(self):
        # how many examples I have
        return len(self.pairs)

    def encode_src(self, text):
        # convert input sentence to numbers
        ids = self.sp.encode(text, out_type=int)
        # if too long, cut it
        ids = ids[: self.max_src]
        return ids

    def encode_tgt(self, text):
        # convert question to numbers
        # I add BOS at start and EOS at end
        # like: <s> question words </s>
        ids = self.sp.encode(text, out_type=int)
        ids = ids[: self.max_tgt - 2]
        ids = [config.BOS_ID] + ids + [config.EOS_ID]
        return ids

    def __getitem__(self, idx):
        # return one example
        src_text, tgt_text = self.pairs[idx]
        src_ids = self.encode_src(src_text)
        tgt_ids = self.encode_tgt(tgt_text)

        item = {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
        return item


def pad_batch(batch):
    """
    In one batch, sentences have different lengths.
    Neural nets like same length, so I add PAD tokens at the end.

    Example:
      [5, 9, 2]     becomes [5, 9, 2, 0, 0]
      [1, 4, 8, 3, 7] stays [1, 4, 8, 3, 7]
    """
    src_list = []
    tgt_list = []
    src_text_list = []
    tgt_text_list = []

    for b in batch:
        src_list.append(b["src"])
        tgt_list.append(b["tgt"])
        src_text_list.append(b["src_text"])
        tgt_text_list.append(b["tgt_text"])

    # lengths before padding
    src_lens = []
    for x in src_list:
        src_lens.append(len(x))
    src_lens = torch.tensor(src_lens, dtype=torch.long)

    tgt_lens = []
    for x in tgt_list:
        tgt_lens.append(len(x))
    tgt_lens = torch.tensor(tgt_lens, dtype=torch.long)

    # pad to same length
    src_pad = torch.nn.utils.rnn.pad_sequence(
        src_list, batch_first=True, padding_value=config.PAD_ID
    )
    tgt_pad = torch.nn.utils.rnn.pad_sequence(
        tgt_list, batch_first=True, padding_value=config.PAD_ID
    )

    return {
        "src": src_pad,
        "tgt": tgt_pad,
        "src_lens": src_lens,
        "tgt_lens": tgt_lens,
        "src_text": src_text_list,
        "tgt_text": tgt_text_list,
    }


def make_loader(tsv_path, shuffle=True, batch_size=None):
    """Make a DataLoader which gives me batches one by one."""
    ds = QGDataset(tsv_path, config.SP_MODEL_FILE)

    if batch_size is None:
        batch_size = config.BATCH_SIZE

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=pad_batch,
        num_workers=0,  # 0 is safer on Windows
    )
    return loader
