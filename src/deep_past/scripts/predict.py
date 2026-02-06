from ..config import Config
from ..pipeline.preprocessor import PreProcessor
from ..pipeline.postprocessor import PostProcessor
from ..stages.predictor import PredictorStage


def main():
    Config.setup()
    PreProcessor(
        in_path=Config.RAW_TEST_FILE,
        out_path=Config.PROCESSED_TEST_FILE,
        write_output=True,
    ).run()
    PredictorStage().run()
    PostProcessor(
        text_column=Config.PROCESSED_TGT_COL,
        write_output=True,
    ).run()


if __name__ == "__main__":
    main()
