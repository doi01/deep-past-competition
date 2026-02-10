import re
import pandas as pd
from tqdm import tqdm

from ..config import Config
from ..log import StepLogger


class PreProcessor:
    """
    Only public method - run().
    If df is provided, it will be used directly and output is not written unless write_output=True.
    """

    def __init__(
        self,
        df=None,
        in_path=None,
        out_path=None,
        verbose=True,
        write_output=False,
    ):
        self.verbose = verbose
        self.in_path = in_path or Config.RAW_TRAIN_FILE
        self.out_path = out_path or Config.PROCESSED_TRAIN_FILE
        self.write_output = write_output
        self.df = df if df is not None else pd.read_csv(self.in_path)
        self.has_tgt = Config.RAW_TGT_COL in self.df.columns
        tqdm.pandas()

    def run(self):
        for step in self._pipeline():
            before = self.df.copy(deep=True)
            self.df = step(self.df)

            if self.verbose:
                col_changes = {
                    col: (before[col], self.df[col])
                    for col in [Config.PROCESSED_SRC_COL, Config.PROCESSED_TGT_COL]
                    if col in before.columns and col in self.df.columns
                }
                StepLogger.log_df_change(step.__name__, before, self.df, col_changes)

        if self.write_output:
            self.df.to_csv(self.out_path, index=False)
        return self.df

    def _pipeline(self):
        steps = [self._clean_columns, self._filter_lengths]
        if self.has_tgt:
            steps.extend([self._filter_ratio, self._drop_duplicates])
        return steps

    @staticmethod
    def _clean_columns(df):
        df = df.copy()
        df[Config.PROCESSED_SRC_COL] = df[Config.RAW_SRC_COL].progress_apply(
            PreProcessor._clean_transliteration
        )
        if Config.RAW_TGT_COL in df.columns:
            df[Config.PROCESSED_TGT_COL] = df[Config.RAW_TGT_COL].progress_apply(
                PreProcessor._clean_translation
            )
        return df

    @staticmethod
    def _clean_transliteration(text):
        text = str(text or "")
        text = re.sub(r"\[\.\.\.\]|\[x+\]|\.{2,}", " <gap> ", text)
        text = re.sub(r"[!?:/\[\]']", "", text)
        text = re.sub(r"[\{\(\)]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _clean_translation(text):
        text = str(text or "")
        text = re.sub(r"\.{2,}|\[.\.\.\]", " <gap> ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _filter_lengths(df):
        mask = (df[Config.PROCESSED_SRC_COL].str.len() > 5) & (
            df[Config.PROCESSED_SRC_COL].str.len() < 1000
        )
        if Config.PROCESSED_TGT_COL in df.columns:
            mask = mask & (df[Config.PROCESSED_TGT_COL].str.len() > 5) & (
                df[Config.PROCESSED_TGT_COL].str.len() < 1200
            )
        return df[mask]

    @staticmethod
    def _filter_ratio(df):
        if Config.PROCESSED_TGT_COL not in df.columns:
            return df
        mask = (
            df[Config.PROCESSED_TGT_COL].str.len()
            / df[Config.PROCESSED_SRC_COL].str.len()
            < 5
        )
        return df[mask]

    @staticmethod
    def _drop_duplicates(df):
        if Config.PROCESSED_TGT_COL not in df.columns:
            return df.drop_duplicates(subset=[Config.PROCESSED_SRC_COL])
        return df.drop_duplicates(
            subset=[Config.PROCESSED_SRC_COL, Config.PROCESSED_TGT_COL]
        )
