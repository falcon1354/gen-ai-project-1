# save_to_drive.py
# ----------------
# This file copies important project results to Google Drive.
# On Colab, /content files can disappear if runtime disconnects.
# Drive files stay safe.
#
# You can also run this anytime:
#   python save_to_drive.py

import os
import shutil
import config


def drive_is_ready():
    # Check if Google Drive is mounted in Colab
    drive_root = "/content/drive/MyDrive"
    if os.path.exists(drive_root):
        return True
    return False


def save_folder_to_drive(local_folder, drive_folder):
    # Copy one local folder into Drive
    if not os.path.exists(local_folder):
        print("  skip (not found yet):", local_folder)
        return

    os.makedirs(drive_folder, exist_ok=True)

    # copy file by file so it is easy to understand
    for name in os.listdir(local_folder):
        src = os.path.join(local_folder, name)
        dst = os.path.join(drive_folder, name)

        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print("  copied file:", src, "->", dst)
        elif os.path.isdir(src):
            # if there is a subfolder, copy whole folder
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print("  copied folder:", src, "->", dst)


def copy_folder_from_drive(drive_folder, local_folder):
    # Copy one folder FROM Drive back to local Colab folder
    if not os.path.exists(drive_folder):
        print("  Drive folder not found:", drive_folder)
        return False

    os.makedirs(local_folder, exist_ok=True)

    for name in os.listdir(drive_folder):
        src = os.path.join(drive_folder, name)
        dst = os.path.join(local_folder, name)

        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print("  restored file:", src, "->", dst)
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print("  restored folder:", src, "->", dst)

    return True


def save_important_files_to_drive():
    """
    Save data / tokenizer / checkpoints into Drive.
    Safe to call many times.
    """
    if config.SAVE_TO_DRIVE == False:
        print("SAVE_TO_DRIVE is False in config.py, so I will not copy to Drive.")
        return

    if drive_is_ready() == False:
        print("Google Drive not found at /content/drive/MyDrive")
        print("If you are on Colab, run drive.mount first.")
        print("If you are on your PC, Drive save is skipped (that is normal).")
        return

    print("Saving important files to Google Drive...")
    print("Drive folder:", config.DRIVE_RESULTS_DIR)

    os.makedirs(config.DRIVE_RESULTS_DIR, exist_ok=True)

    # 1) data
    save_folder_to_drive(
        config.DATA_DIR,
        os.path.join(config.DRIVE_RESULTS_DIR, "data"),
    )

    # 2) tokenizer
    save_folder_to_drive(
        config.TOKENIZER_DIR,
        os.path.join(config.DRIVE_RESULTS_DIR, "tokenizer"),
    )

    # 3) checkpoints (most important for resume)
    save_folder_to_drive(
        config.CHECKPOINT_DIR,
        os.path.join(config.DRIVE_RESULTS_DIR, "checkpoints"),
    )

    print("Done saving to Drive :)")


def restore_important_files_from_drive():
    """
    If Colab restarted and local files are gone,
    copy them back from Google Drive.
    """
    if drive_is_ready() == False:
        print("Drive not mounted, cannot restore.")
        return False

    print("Trying to restore files from Google Drive...")
    print("Drive folder:", config.DRIVE_RESULTS_DIR)

    if not os.path.exists(config.DRIVE_RESULTS_DIR):
        print("No saved results folder found on Drive yet.")
        return False

    ok1 = copy_folder_from_drive(
        os.path.join(config.DRIVE_RESULTS_DIR, "data"),
        config.DATA_DIR,
    )
    ok2 = copy_folder_from_drive(
        os.path.join(config.DRIVE_RESULTS_DIR, "tokenizer"),
        config.TOKENIZER_DIR,
    )
    ok3 = copy_folder_from_drive(
        os.path.join(config.DRIVE_RESULTS_DIR, "checkpoints"),
        config.CHECKPOINT_DIR,
    )

    if ok1 or ok2 or ok3:
        print("Restore finished.")
        return True

    print("Nothing restored.")
    return False


if __name__ == "__main__":
    save_important_files_to_drive()
