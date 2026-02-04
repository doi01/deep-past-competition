import re
import pandas as pd
from tqdm import tqdm
from .config import Config


def clean_transliteration(text: str) -> str:
    if pd.isna(text): return ""
    text = str(text)

    text = re.sub(r"\[\.\.\.\]|\[… …\]|\[x+\]|\.{2,}|…", " <gap> ", text)
    text = re.sub(r"[!?:/˹˺\[\]']", "", text)
    text = re.sub(r"\((.*?)\)", r"\1", text)
    text = text.replace("Ḫ", "H").replace("ḫ", "h")
    subscripts = {"₀":"0","₁":"1","₂":"2","₃":"3","₄":"4","₅":"5","₆":"6","₇":"7","₈":"8","₉":"9","ₓ":"x"}
    for k, v in subscripts.items():
        text = text.replace(k, v)

    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_translation(text: str) -> str:
    if pd.isna(text): return ""
    text = str(text)
    text = re.sub(r"\.{2,}|…|\[…\]", " <gap> ", text)
    text = re.sub(r"[˹˺]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_pipeline():
    """
    Полный пайплайн: Чтение -> Выравнивание -> Очистка -> Сохранение
    """
    print(f"Reading from {Config.RAW_DATA_PATH}...")
    df = pd.read_csv(Config.RAW_DATA_PATH)
    original_len = len(df)
    
    # 1. Cleaning
    tqdm.pandas(desc="Cleaning SRC")
    df['src'] = df['transliteration'].progress_apply(clean_transliteration)
    
    tqdm.pandas(desc="Cleaning TGT")
    df['tgt'] = df['translation'].progress_apply(clean_translation)

    # 2. Post-cleaning filtering
    # Удаляем пустые строки или строки, ставшие пустыми после очистки
    df = df[df['src'].str.len() > 1]
    df = df[df['tgt'].str.len() > 1]
    
    # Удаляем дубликаты
    df = df.drop_duplicates(subset=['src', 'tgt'])
    
    print(f"Rows after cleaning & dedup: {len(df)}")

    # 3. Saving
    Config.setup() # Создаем папки
    save_cols = ['src', 'tgt']
    df[save_cols].to_csv(Config.PROCESSED_DATA_PATH, index=False)
    print(f"Saved to {Config.PROCESSED_DATA_PATH}")
    
    return df[save_cols]