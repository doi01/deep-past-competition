import re
import pandas as pd
from tqdm import tqdm

try:
    from .config import Config
except ImportError:
    from config import Config

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

def simple_sentence_splitter(text, max_length=200):
    if pd.isna(text):
        return []

    separators = [
        "\n",
        "  ",
    ]

    sentences = [text]
    for sep in separators:
        new_sentences = []
        for sent in sentences:
            new_sentences.extend(sent.split(sep))
        sentences = new_sentences

    sentences = [s.strip() for s in sentences if s.strip()]

    final_sentences = []
    for sent in sentences:
        if len(sent) <= max_length:
            final_sentences.append(sent)
        else:
            words = sent.split()
            current = []
            for word in words:
                current.append(word)
                if len(" ".join(current)) > max_length:
                    final_sentences.append(" ".join(current[:-1]))
                    current = [word]
            if current:
                final_sentences.append(" ".join(current))

    return final_sentences

def simple_sentence_aligner(
    df: pd.DataFrame,
    score_threshold: float | None = None,
    auto_threshold: bool = True,
    sample_size: int = 400,
    random_seed: int = 13,
    log_stats: bool = True,
) -> pd.DataFrame:
    def split_src_parts(text: str) -> list[str]:
        return simple_sentence_splitter(text)

    def split_tgt_sents(text: str) -> list[str]:
        return [t.strip() for t in re.split(r"(?<=[.!?])\s+", text) if t.strip()]

    def normalize_ascii(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union

    def seq_ratio(a: str, b: str) -> float:
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()

    def digit_set(s: str) -> set:
        return set(re.findall(r"\d+", s))

    def length_score(a: str, b: str) -> float:
        la = len(a)
        lb = len(b)
        if la == 0 or lb == 0:
            return 0.0
        return 1.0 - abs(la - lb) / max(la, lb)

    def similarity(a: str, b: str) -> float:
        lscore = length_score(a, b)
        dscore = jaccard(digit_set(a), digit_set(b))
        seq = seq_ratio(normalize_ascii(a), normalize_ascii(b))
        return 0.7 * lscore + 0.2 * dscore + 0.1 * seq

    def align_by_similarity(src_parts: list[str], tgt_sents: list[str], max_merge: int = 3):
        n = len(src_parts)
        m = len(tgt_sents)
        dp = [[-1e9] * (m + 1) for _ in range(n + 1)]
        back = [[None] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0.0

        for i in range(n + 1):
            for j in range(m + 1):
                if dp[i][j] <= -1e8:
                    continue
                for a in range(1, max_merge + 1):
                    for b in range(1, max_merge + 1):
                        if i + a > n or j + b > m:
                            continue
                        src_chunk = " ".join(src_parts[i:i + a])
                        tgt_chunk = " ".join(tgt_sents[j:j + b])
                        score = similarity(src_chunk, tgt_chunk)
                        score_adj = score - 0.02 * (a + b - 2)
                        cand = dp[i][j] + score_adj
                        if cand > dp[i + a][j + b]:
                            dp[i + a][j + b] = cand
                            back[i + a][j + b] = (i, j, a, b, score)

        if dp[n][m] <= -1e8:
            return [], 0.0

        pairs = []
        i, j = n, m
        total_score = 0.0
        steps = 0
        while i > 0 and j > 0:
            prev = back[i][j]
            if not prev:
                return [], 0.0
            pi, pj, a, b, score = prev
            src_chunk = " ".join(src_parts[pi:pi + a])
            tgt_chunk = " ".join(tgt_sents[pj:pj + b])
            pairs.append((src_chunk, tgt_chunk, score))
            total_score += score
            steps += 1
            i, j = pi, pj
        pairs.reverse()
        avg_score = total_score / max(steps, 1)
        return pairs, avg_score

    def pick_threshold(sample_scores: list[float]) -> float:
        if not sample_scores:
            return 0.28
        sample_scores = sorted(sample_scores)
        idx = int(0.5 * (len(sample_scores) - 1))
        return min(0.6, max(0.25, sample_scores[idx]))

    if score_threshold is None and auto_threshold:
        import random
        candidates = []
        for idx, row in df.iterrows():
            src = str(row.get("transliteration", ""))
            tgt = str(row.get("translation", ""))
            src_parts = split_src_parts(src)
            tgt_sents = split_tgt_sents(tgt)
            if len(src_parts) >= 2 and len(tgt_sents) >= 2 and len(src_parts) != len(tgt_sents):
                candidates.append((idx, src_parts, tgt_sents))
        random.Random(random_seed).shuffle(candidates)
        candidates = candidates[:sample_size]
        scores = []
        for _, src_parts, tgt_sents in candidates:
            max_merge = 4 if abs(len(src_parts) - len(tgt_sents)) >= 2 else 3
            pairs, avg_score = align_by_similarity(src_parts, tgt_sents, max_merge=max_merge)
            if pairs:
                scores.append(avg_score)
        score_threshold = pick_threshold(scores)
        if log_stats:
            print(f"[align] auto score_threshold: {score_threshold:.3f} (samples={len(scores)})")
    elif score_threshold is None:
        score_threshold = 0.28

    aligned_data = []
    stats = {
        "exact_1to1": 0,
        "similarity_used": 0,
        "similarity_rejected": 0,
        "fallback_no_split": 0,
        "fallback_after_sim": 0,
    }

    for _, row in df.iterrows():
        src = str(row.get("transliteration", ""))
        tgt = str(row.get("translation", ""))

        tgt_sents = split_tgt_sents(tgt)
        src_parts = split_src_parts(src)

        if len(src_parts) < 2 and len(tgt_sents) < 2:
            aligned_data.append({"transliteration": src, "translation": tgt})
            stats["fallback_no_split"] += 1
            continue

        if len(src_parts) >= 2 and len(tgt_sents) >= 2 and len(src_parts) == len(tgt_sents):
            for s, t in zip(src_parts, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned_data.append({"transliteration": s, "translation": t})
            stats["exact_1to1"] += 1
            continue

        if len(src_parts) >= 2 and len(tgt_sents) >= 2:
            max_merge = 4 if abs(len(src_parts) - len(tgt_sents)) >= 2 else 3
            pairs, avg_score = align_by_similarity(src_parts, tgt_sents, max_merge=max_merge)
            if pairs and avg_score >= score_threshold:
                for s, t, _ in pairs:
                    if len(s) > 3 and len(t) > 3:
                        aligned_data.append({"transliteration": s, "translation": t})
                stats["similarity_used"] += 1
                continue
            stats["similarity_rejected"] += 1

        aligned_data.append({"transliteration": src, "translation": tgt})
        stats["fallback_after_sim"] += 1

    if log_stats:
        print(
            "[align] stats "
            f"exact_1to1={stats['exact_1to1']}, "
            f"similarity_used={stats['similarity_used']}, "
            f"similarity_rejected={stats['similarity_rejected']}, "
            f"fallback_no_split={stats['fallback_no_split']}, "
            f"fallback_after_sim={stats['fallback_after_sim']}"
        )

    return pd.DataFrame(aligned_data)

def preprocess_pipeline():
    print(f"Starting advanced preprocessing from {Config.RAW_TRAIN_FILE}...")
    df = pd.read_csv(Config.RAW_TRAIN_FILE)

    df = simple_sentence_aligner(df)
    
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
    df[['src', 'tgt']].to_csv(Config.PROCESSED_TRAIN_FILE, index=False)
    
    print("\nCheck Max Lengths:")
    print(f"Max SRC: {df['src'].str.len().max()} | Max TGT: {df['tgt'].str.len().max()}")
    
    return df
