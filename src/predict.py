# Здесь просто делаем предсказание, обучение в блокноте training.ipynb
# Если будешь обучать - просто скопируй в кагл/коллаб из training.ipynb

import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

from tqdm import tqdm
from config import Config

MODEL_DIR = Config.MODEL_DIR
TEST_FILE = Config.TEST_FILE
OUTPUT_FILE = Config.PREDICTIONS_FILE

# ---------------------------
# Загрузка модели и токенизатора
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
model.eval()

# ---------------------------
# Загрузка теста
# ---------------------------
df_test = pd.read_csv(TEST_FILE)
if "src" not in df_test.columns and "transliteration" not in df_test.columns:
    raise ValueError("В тестовом CSV должна быть колонка 'src' или 'transliteration'")

text_column = "src" if "src" in df_test.columns else "transliteration"
texts = df_test[text_column].tolist()
print(f"Всего строк для перевода: {len(texts)}")

# ---------------------------
# Генерация предсказаний
# ---------------------------
BATCH_SIZE = 8  # подбирай под память GPU/CPU
preds = []

for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = ["translate Akkadian to English: " + t for t in texts[i:i+BATCH_SIZE]]
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    preds.extend(decoded)

# ---------------------------
# Сохранение предсказаний
# ---------------------------
submission = pd.DataFrame({
    "id": df_test["id"],          # берем id из исходного файла
    "translation": preds          # берем предсказания
})

submission.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Submission saved to {OUTPUT_FILE}")

