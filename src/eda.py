import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_stats(series: pd.Series, name: str):
    lengths = series.astype(str).str.len()
    return {
        f"{name}_mean": lengths.mean(),
        f"{name}_median": lengths.median(),
        f"{name}_95%": np.percentile(lengths, 95),
        f"{name}_max": lengths.max()
    }

def plot_distributions(df: pd.DataFrame, src_col: str, tgt_col: str, title: str):
    """Рисует гистограммы длин"""
    src_len = df[src_col].astype(str).str.len()
    tgt_len = df[tgt_col].astype(str).str.len()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{title} - Length Distribution', fontsize=16)

    sns.histplot(src_len, bins=50, ax=axes[0], color='skyblue', kde=True)
    axes[0].set_title(f'Source Length ({src_col})')
    axes[0].set_xlabel('Characters')
    
    sns.histplot(tgt_len, bins=50, ax=axes[1], color='orange', kde=True)
    axes[1].set_title(f'Target Length ({tgt_col})')
    axes[1].set_xlabel('Characters')
    
    plt.tight_layout()
    plt.show()

def check_vocab(df: pd.DataFrame, col: str, top_n=10):
    """Анализирует уникальные символы"""
    text = "".join(df[col].astype(str).tolist())
    unique_chars = sorted(list(set(text)))
    print(f"[{col}] Unique characters count: {len(unique_chars)}")
    print(f"[{col}] Sample chars: {unique_chars[:10]} ... {unique_chars[-10:]}")

def analyze_dataset(df: pd.DataFrame, src_col='transliteration', tgt_col='translation', name="Dataset"):
    """
    Главная функция для вызова в ноутбуке.
    """
    print(f"--- Analyzing: {name} ---")
    print(f"Total samples: {len(df)}")
    
    # Stats
    stats = {}
    stats.update(get_stats(df[src_col], "SRC"))
    stats.update(get_stats(df[tgt_col], "TGT"))
    
    stats_df = pd.DataFrame([stats])
    print("\nLength Statistics:")
    print(stats_df.round(2).to_string(index=False))
    
    # Vocab check
    print("\nVocabulary Check:")
    check_vocab(df, src_col)
    check_vocab(df, tgt_col)
    
    # Plots
    plot_distributions(df, src_col, tgt_col, name)