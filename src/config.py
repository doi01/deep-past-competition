import os

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    MODEL_NAME = 'google/byt5-small'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 2e-4
    SEED = 42
    
    # Полные пути
    RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'train.csv')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, 'train_preprocessed.csv')
    SENTENS_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'Sentences_Oare_FirstWord_LinNum.csv')
    
    OUTPUT_DIR = os.path.join(BASE_DIR, 'byt5-akkadian-model')

    @staticmethod
    def setup():
        """Создает необходимые папки, если их нет"""
        os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        print(f"Directory check complete. Data path: {Config.PROCESSED_DATA_PATH}")