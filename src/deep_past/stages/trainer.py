import gc
import numpy as np
import pandas as pd
import torch
import evaluate

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from ..config import Config
from ..seed import seed_everything
from .callbacks import DeleteOldCheckpointsCallback


class TrainerStage:
    """
    Only public method - run().
    """

    def run(self):
        Config.setup()
        seed_everything(Config.SEED)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        df = pd.read_csv(Config.PROCESSED_TRAIN_FILE)
        required_cols = {Config.PROCESSED_SRC_COL, Config.PROCESSED_TGT_COL}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain columns {required_cols}")

        dataset = Dataset.from_pandas(df)
        split = dataset.train_test_split(test_size=0.1, seed=Config.SEED)

        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        prefix = "translate Akkadian to English: "

        def tokenize(batch):
            inputs = [prefix + x for x in batch[Config.PROCESSED_SRC_COL]]
            targets = batch[Config.PROCESSED_TGT_COL]

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

        train_ds = split["train"].map(
            tokenize, batched=True, remove_columns=split["train"].column_names
        )
        val_ds = split["test"].map(
            tokenize, batched=True, remove_columns=split["test"].column_names
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)
        model.to(device)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
        )

        metric = evaluate.load("chrf")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            preds = preds[0] if isinstance(preds, tuple) else preds

            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

            labels = np.where(
                labels != -100,
                labels,
                tokenizer.pad_token_id,
            )
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_labels = [[l.strip()] for l in decoded_labels]
            decoded_preds = [p.strip() for p in decoded_preds]

            score = metric.compute(
                predictions=decoded_preds,
                references=decoded_labels,
            )["score"]

            return {"chrf": score}

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
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[DeleteOldCheckpointsCallback()],
        )

        trainer.train()
        trainer.save_model(Config.OUTPUT_DIR)
        tokenizer.save_pretrained(Config.OUTPUT_DIR)
