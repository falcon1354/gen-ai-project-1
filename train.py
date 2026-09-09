# train.py
# --------
# STEP 3
# This file trains the model.
#
# Simple training idea:
# 1) give model input sentence
# 2) model tries to make question
# 3) compare with real question (loss)
# 4) update model weights a little
# 5) repeat many times
#
# RESUME SUPPORT:
# If training stops (Colab disconnect), run train.py again.
# It will load the last checkpoint and continue from the next epoch.
# Checkpoints are also copied to Google Drive for safety.

import os
import math
import torch
import torch.nn as nn
from tqdm import tqdm
import sentencepiece as spm

import config
from dataset import make_loader
from model import Seq2Seq
from save_to_drive import save_important_files_to_drive, restore_important_files_from_drive


def get_device():
    """Use GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """
    Run through the dataset one time.
    If train=True, we update weights.
    If train=False, we only check validation loss.
    """
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0

    if train:
        desc = "train"
    else:
        desc = "valid"

    loop = tqdm(loader, desc=desc, leave=False)

    for batch in loop:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_lens = batch["src_lens"].to(device)

        if train:
            optimizer.zero_grad()

        # turn off gradient math during validation (faster / safer)
        if train:
            outputs = model(
                src,
                src_lens,
                tgt,
                teacher_forcing_ratio=config.TEACHER_FORCING_RATIO,
            )
        else:
            with torch.no_grad():
                outputs = model(
                    src,
                    src_lens,
                    tgt,
                    teacher_forcing_ratio=0.0,
                )

        # we dont predict position 0 (<s>), we start from position 1
        logits = outputs[:, 1:]
        gold = tgt[:, 1:]

        # flatten for loss function
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            gold.reshape(-1),
        )

        if train:
            loss.backward()
            # clip gradients so training stays stable
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)
            optimizer.step()

        # average by real tokens (ignore pad)
        n_tokens = (gold != config.PAD_ID).sum().item()
        total_loss = total_loss + (loss.item() * n_tokens)
        total_tokens = total_tokens + n_tokens
        loop.set_postfix(loss=loss.item())

    avg = total_loss / max(total_tokens, 1)

    # perplexity = how confused the model is (lower is better)
    # I cap avg so math.exp doesnt crash
    if avg > 20:
        ppl = math.exp(20)
    else:
        ppl = math.exp(avg)

    return avg, ppl


def save_checkpoint(path, model, optimizer, epoch, best_val, vocab_size):
    # Save everything needed to continue training later
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "vocab_size": vocab_size,
        },
        path,
    )


def try_load_resume_checkpoint(model, optimizer, device):
    """
    Try to load last training progress.
    Returns: start_epoch, best_val, loaded_or_not
    """
    last_path = os.path.join(config.CHECKPOINT_DIR, "last_checkpoint.pt")
    drive_last = os.path.join(
        config.DRIVE_RESULTS_DIR, "checkpoints", "last_checkpoint.pt"
    )

    # If local checkpoint missing, try restore from Drive first
    if not os.path.exists(last_path):
        print("No local last_checkpoint.pt found.")
        print("Checking Google Drive backup...")
        restore_important_files_from_drive()

    # still missing?
    if not os.path.exists(last_path):
        # maybe only Drive has it and restore failed path-wise; try direct copy
        if os.path.exists(drive_last):
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            import shutil
            shutil.copy2(drive_last, last_path)
            print("Copied last_checkpoint.pt from Drive.")
        else:
            print("No checkpoint found. Starting training from epoch 1.")
            return 1, 999999.0, False

    print("Found checkpoint:", last_path)
    saved = torch.load(last_path, map_location=device)

    model.load_state_dict(saved["model_state"])

    # older checkpoints might not have optimizer_state
    if "optimizer_state" in saved:
        optimizer.load_state_dict(saved["optimizer_state"])

    finished_epoch = saved.get("epoch", 0)
    best_val = saved.get("best_val", saved.get("val_loss", 999999.0))

    # continue from NEXT epoch
    start_epoch = finished_epoch + 1

    print("Resuming from epoch:", start_epoch)
    print("Best validation loss so far:", best_val)
    return start_epoch, best_val, True


def main():
    # If Colab restarted, restore data/tokenizer/checkpoints from Drive
    if not os.path.exists(config.TRAIN_TSV) or not os.path.exists(config.SP_MODEL_FILE):
        print("Local data/tokenizer missing. Trying Drive restore...")
        restore_important_files_from_drive()

    # make sure previous steps are done
    needed = [config.TRAIN_TSV, config.VALID_TSV, config.SP_MODEL_FILE]
    for path in needed:
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Missing " + path + ". Run prepare_data.py and train_tokenizer.py first."
            )

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    device = get_device()
    print("using device:", device)

    # vocab size from tokenizer
    sp = spm.SentencePieceProcessor(model_file=config.SP_MODEL_FILE)
    vocab_size = sp.get_piece_size()
    print("vocab size:", vocab_size)

    train_loader = make_loader(config.TRAIN_TSV, shuffle=True)
    valid_loader = make_loader(config.VALID_TSV, shuffle=False)

    model = Seq2Seq(vocab_size).to(device)

    # ignore_index means: dont calculate loss on PAD tokens
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # try resume
    start_epoch, best_val, resumed = try_load_resume_checkpoint(model, optimizer, device)

    last_path = os.path.join(config.CHECKPOINT_DIR, "last_checkpoint.pt")
    best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")

    # if already finished all epochs, stop
    if start_epoch > config.NUM_EPOCHS:
        print("Training already completed all", config.NUM_EPOCHS, "epochs.")
        print("No need to train more.")
        return

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        print("\n=== Epoch", epoch, "/", config.NUM_EPOCHS, "===")

        train_loss, train_ppl = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_ppl = run_epoch(
            model, valid_loader, criterion, optimizer, device, train=False
        )

        print(
            "train loss", round(train_loss, 4),
            "(ppl", round(train_ppl, 2), ") |",
            "valid loss", round(val_loss, 4),
            "(ppl", round(val_ppl, 2), ")"
        )

        # ALWAYS save last checkpoint after each full epoch
        # this is what we use to resume later
        save_checkpoint(last_path, model, optimizer, epoch, best_val, vocab_size)
        print("  saved resume checkpoint to", last_path)

        # ALSO save best model if validation improved
        if val_loss < best_val:
            best_val = val_loss
            # update best_val inside last checkpoint too
            save_checkpoint(last_path, model, optimizer, epoch, best_val, vocab_size)

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab_size": vocab_size,
                    "val_loss": val_loss,
                    "best_val": best_val,
                    "epoch": epoch,
                },
                best_path,
            )
            print("  saved new best model to", best_path)

        # copy everything to Google Drive after every epoch
        # so even if Colab dies, we can continue later
        print("  saving progress to Google Drive...")
        save_important_files_to_drive()

    print("\nTraining finished. Best valid loss:", best_val)

    # final backup to Drive
    print("\nFinal save to Google Drive...")
    save_important_files_to_drive()


if __name__ == "__main__":
    main()
