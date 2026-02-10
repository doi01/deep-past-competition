import glob
import os
import shutil
from transformers import TrainerCallback


class DeleteOldCheckpointsCallback(TrainerCallback):
    def __init__(self, keep_last=2):
        self.keep_last = keep_last

    def on_save(self, args, state, control, **kwargs):
        checkpoints = sorted(
            glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
            key=os.path.getmtime,
        )
        for ckpt in checkpoints[:-self.keep_last]:
            shutil.rmtree(ckpt, ignore_errors=True)
