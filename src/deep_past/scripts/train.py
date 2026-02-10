from ..config import Config
from ..pipeline.preprocessor import PreProcessor
from ..stages.trainer import TrainerStage


def main():
    Config.setup()
    PreProcessor(write_output=True).run()
    TrainerStage().run()


if __name__ == "__main__":
    main()
