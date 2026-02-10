import os
import gc
import glob
import shutil
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from .preprocessing import preprocess_pipeline
import evaluate

from config import Config
from seed import seed_everything


# ---------------------------
# Callbacks
# ---------------------------
class DeleteOldCheckpointsCallback(TrainerCallback):
    """
    Оставляет только последние N чекпоинтов
    """
    def __init__(self, keep_last=2):
        self.keep_last = keep_last

    def on_save(self, args, state, control, **kwargs):
        checkpoints = sorted(
            glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
            key=os.path.getmtime
        )
        for ckpt in checkpoints[:-self.keep_last]:
            shutil.rmtree(ckpt, ignore_errors=True)
            print(f"Deleted old checkpoint: {ckpt}")


# ---------------------------
# Main
# ---------------------------
def main():
    # --------- setup ---------
    Config.setup()
    seed_everything(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # --------- load data ---------
    print("Loading dataset:", Config.RAW_TRAIN_FILE)
    df = preprocess_pipeline()


    required_cols = {"src", "tgt"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required_cols}")

    dataset = Dataset.from_pandas(df)
    split = dataset.train_test_split(test_size=0.1, seed=Config.SEED)

    train_ds = split["train"]
    val_ds = split["test"]

    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    # --------- tokenizer ---------
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    PREFIX = "translate Akkadian to English: "

    def tokenize(batch):
        inputs = [PREFIX + x for x in batch["src"]]
        targets = batch["tgt"]

        model_inputs = tokenizer(
            inputs,
            truncation=True,
            max_length=Config.MAX_LENGTH,
        )
        labels = tokenizer(
            targets,
            truncation=True,
            max_length=Config.MAX_LENGTH,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(
        tokenize,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    tokenized_val = val_ds.map(
        tokenize,
        batched=True,
        remove_columns=val_ds.column_names,
    )

    gc.collect()
    torch.cuda.empty_cache()

    # --------- model ---------
    model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)
    model.to(device)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    # --------- metric ---------
    metric = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True
        )

        labels = np.where(
            labels != -100,
            labels,
            tokenizer.pad_token_id,
        )
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]

        score = metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )["score"]

        return {"chrf": score}

    # --------- training args ---------
    training_args = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",

        logging_steps=10,
        eval_steps=50,
        save_steps=50,

        save_total_limit=2,

        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,

        gradient_accumulation_steps=4,
        num_train_epochs=Config.EPOCHS,

        fp16=torch.cuda.is_available(),
        predict_with_generate=True,

        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[DeleteOldCheckpointsCallback()],
    )

    # --------- train ---------
    print("Starting training...")
    trainer.train()

    # --------- save ---------
    print("Saving final model...")
    trainer.save_model(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)

    print("Training complete")


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    main()
