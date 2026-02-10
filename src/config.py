import os

class Config:
    IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

    BASE_DIR = (
        "/kaggle/working"
        if IS_KAGGLE
        else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    MODEL_NAME = "google/byt5-base"
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 2e-4
    SEED = 42
 
    if IS_KAGGLE:
        DATA_DIR = "/kaggle/input/deep-past-initiative-machine-translation"
        MODEL_DIR = "/kaggle/input/v1/pytorch/default/1"
        OUTPUT_DIR = "/kaggle/working"
    else:
        DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
        MODEL_DIR = os.path.join(BASE_DIR, "models", "byt5-akkadian-model")
        OUTPUT_DIR = os.path.join(BASE_DIR, "models", "byt5-akkadian-model")

    RAW_TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
    TEST_FILE = os.path.join(DATA_DIR, "test.csv")

    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    PROCESSED_TRAIN_FILE = os.path.join(PROCESSED_DIR, "train_preprocessed.csv")

    PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "submission.csv")

    SENTENCES_DATA_PATH = os.path.join(
        BASE_DIR, "data", "raw", "Sentences_Oare_FirstWord_LinNum.csv"
    )

    # Column name constants
    RAW_SRC_COL = 'transliteration'
    RAW_TGT_COL = 'translation'
    PROCESSED_SRC_COL = 'src'
    PROCESSED_TGT_COL = 'tgt'

    @staticmethod
    def setup():
        if not Config.IS_KAGGLE:
            os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        print("Environment:", "Kaggle" if Config.IS_KAGGLE else "Local")
        print("BASE_DIR:", Config.BASE_DIR)
        print("MODEL_DIR:", Config.MODEL_DIR)
        print("DATA_DIR:", Config.DATA_DIR)
