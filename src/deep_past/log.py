import logging


class StepLogger:
    _logger = None

    @classmethod
    def get_logger(cls, logfile="pipeline.log"):
        if cls._logger is None:
            cls._logger = logging.getLogger("PipelineLogger")
            cls._logger.setLevel(logging.INFO)
            fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            fh.setFormatter(formatter)
            cls._logger.addHandler(fh)
        return cls._logger

    @classmethod
    def log_df_change(cls, step_name, before_df, after_df, col_changes=None):
        logger = cls.get_logger()
        removed_rows = len(before_df) - len(after_df)
        logger.info(
            f"[{step_name}] Rows before: {len(before_df)}, after: {len(after_df)}, removed: {removed_rows}"
        )
        if col_changes:
            for col, (before_col, after_col) in col_changes.items():
                total_before = before_col.str.len().sum()
                total_after = after_col.str.len().sum()
                diff = total_before - total_after
                logger.info(
                    f"[{step_name}] Column '{col}' chars: {total_before} -> {total_after}, diff: {diff}"
                )
