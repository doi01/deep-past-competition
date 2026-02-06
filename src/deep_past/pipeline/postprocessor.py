import re
import pandas as pd
from tqdm import tqdm

from ..config import Config
from ..log import StepLogger


class PostProcessor:
    """
    Only public method - run().
    If df is provided, it will be used directly and output is not written unless write_output=True.
    """

    def __init__(
        self,
        df=None,
        text_column=Config.PROCESSED_TGT_COL,
        in_path=None,
        out_path=None,
        verbose=True,
        write_output=False,
    ):
        self.verbose = verbose
        self.text_column = text_column
        self.in_path = in_path or Config.BEFORE_POSTPROCESS_FILE
        self.out_path = out_path or Config.SUBMISSION_FILE
        self.write_output = write_output
        self.df = df if df is not None else pd.read_csv(self.in_path)
        tqdm.pandas()

    def run(self):
        for step in self._pipeline():
            before = self.df[self.text_column].copy(deep=True)
            self.df[self.text_column] = self.df[self.text_column].progress_apply(step)

            if self.verbose:
                col_changes = {self.text_column: (before, self.df[self.text_column])}
                StepLogger.log_df_change(
                    step.__name__,
                    before_df=before.to_frame(),
                    after_df=self.df[[self.text_column]],
                    col_changes=col_changes,
                )

        if self.write_output:
            self.df.to_csv(self.out_path, index=False)
        return self.df

    def _pipeline(self):
        return [
            self._normalize_whitespace,
            self._remove_annotations,
            self._normalize_gaps,
            self._remove_forbidden_symbols,
            self._deduplicate_words,
            self._deduplicate_ngrams,
            self._trim,
        ]

    @staticmethod
    def _normalize_whitespace(text):
        text = (text or "").replace("\n", " ")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _remove_annotations(text):
        return re.sub(r"\([^)]*\)", "", text or "")

    @staticmethod
    def _normalize_gaps(text):
        patterns = [r"\[\.\.\.\]", r"\.\.\.", r"\[x\]", r"x{2,}", r"\<gap\>"]
        for p in patterns:
            text = re.sub(p, " <gap> ", text or "", flags=re.IGNORECASE)
        return text

    @staticmethod
    def _remove_forbidden_symbols(text):
        forbidden = "*"
        for ch in forbidden:
            text = (text or "").replace(ch, "")
        return text

    @staticmethod
    def _deduplicate_words(text):
        words = (text or "").split()
        cleaned, prev = [], None
        for w in words:
            if w != prev:
                cleaned.append(w)
            prev = w
        return " ".join(cleaned)

    @staticmethod
    def _deduplicate_ngrams(text, n=3):
        words = (text or "").split()
        result, i = [], 0
        while i < len(words):
            chunk = words[i : i + n]
            next_chunk = words[i + n : i + 2 * n]
            if chunk == next_chunk:
                result.extend(chunk)
                i += 2 * n
            else:
                result.append(words[i])
                i += 1
        return " ".join(result)

    @staticmethod
    def _trim(text):
        return (text or "").strip(" .,-;:")
