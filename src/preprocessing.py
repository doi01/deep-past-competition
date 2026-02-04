import re
import pandas as pd
from tqdm import tqdm
from .config import Config

def clean_transliteration(text: str) -> str:
    if pd.isna(text): return ""
    text = str(text)

    # 1. Лакуны и пропуски
    text = re.sub(r"\[\.\.\.\]|\[… …\]|\[x+\]|\.{2,}|…", " <gap> ", text)
    
    # 2. Удаление спецсимволов разметки
    text = re.sub(r"[!?:/˹˺\[\]'⌈⌉]", "", text)
    
    # 3. Скобки: (word) -> word
    text = re.sub(r"[\{\(\)]", "", text)
    
    # 4. Фонетическая нормализация
    text = text.replace("Ḫ", "H").replace("ḫ", "h")
    
    # 5. Подстрочные и надстрочные знаки в обычные цифры
    subscripts = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")
    text = text.translate(subscripts)

    # 6. Финальная чистка пробелов
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_translation(text: str) -> str:
    if pd.isna(text): return ""
    text = str(text)

    # 1. НОРМАЛИЗАЦИЯ ДРОБЕЙ
    fractions = {
        "⅓": "1/3", "⅔": "2/3", "⅚": "5/6", "⅛": "1/8", 
        "¼": "1/4", "½": "1/2", "¾": "3/4"
    }
    for k, v in fractions.items():
        text = text.replace(k, v)

    # 2. Типографика: Кавычки и тире
    text = re.sub(r"[“”„«»]", '"', text)
    text = re.sub(r"[–—]", "-", text)
    
    # 3. Пропуски
    text = re.sub(r"\.{2,}|…|\[…\]", " <gap> ", text)
    
    # 4. Скобки и мусор
    text = re.sub(r"[˹˺\[\]]", "", text)
    
    # 5. Финальная чистка пробелов
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_pipeline():
    print(f"Starting advanced preprocessing from {Config.RAW_DATA_PATH}...")
    df = pd.read_csv(Config.RAW_DATA_PATH)
    
    tqdm.pandas(desc="Cleaning Transliteration")
    df['src'] = df['transliteration'].progress_apply(clean_transliteration)
    
    tqdm.pandas(desc="Cleaning Translation")
    df['tgt'] = df['translation'].progress_apply(clean_translation)

    before_count = len(df)
    
    df = df[
        (df['src'].str.len() > 5) & (df['src'].str.len() < 1000) &
        (df['tgt'].str.len() > 5) & (df['tgt'].str.len() < 1200)
    ]

    df = df[df['tgt'].str.len() / df['src'].str.len() < 5]
    
    df = df.drop_duplicates(subset=['src', 'tgt'])
    
    after_count = len(df)
    print(f"Removed {before_count - after_count} anomalous rows (tails/duplicates).")
    print(f"Final dataset size: {after_count} samples.")

    Config.setup()
    df[['src', 'tgt']].to_csv(Config.PROCESSED_DATA_PATH, index=False)
    
    print("\nCheck Max Lengths:")
    print(f"Max SRC: {df['src'].str.len().max()} | Max TGT: {df['tgt'].str.len().max()}")
    
    return df