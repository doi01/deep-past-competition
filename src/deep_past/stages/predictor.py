import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from ..config import Config


class PredictorStage:
    """
    Only public method - run().
    """

    def run(self):
        df = pd.read_csv(Config.PROCESSED_TEST_FILE)
        if Config.PROCESSED_SRC_COL not in df.columns:
            raise ValueError(
                f"Missing column '{Config.PROCESSED_SRC_COL}' in {Config.PROCESSED_TEST_FILE}"
            )

        tokenizer = AutoTokenizer.from_pretrained(Config.OUTPUT_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(Config.OUTPUT_DIR)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        inputs = tokenizer(
            df[Config.PROCESSED_SRC_COL].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=Config.MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=Config.MAX_LENGTH)

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        out_df = pd.DataFrame(
            {
                "id": df.get("id", pd.Series(range(len(df)))),
                Config.PROCESSED_TGT_COL: preds,
            }
        )

        out_df.to_csv(Config.BEFORE_POSTPROCESS_FILE, index=False)
        return out_df
