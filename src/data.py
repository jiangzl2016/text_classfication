"""把training.csv按照 80/20 分成训练集和测试集, 存成data/train.csv和data/test.csv."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "NLP-文本分类" / "training.csv"
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, header=None, names=["label", "text"], dtype={"label": int, "text": str})
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df


def split_and_persist() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_raw()
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    return train_df, test_df


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        return split_and_persist()
    train_df = pd.read_csv(TRAIN_CSV, dtype={"label": int, "text": str})
    test_df = pd.read_csv(TEST_CSV, dtype={"label": int, "text": str})
    return train_df, test_df


def get_kfold() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


if __name__ == "__main__":
    train_df, test_df = split_and_persist()
    print(f"原始数据行数      : {len(train_df) + len(test_df)}")
    print(f"训练集行数 (80%)  : {len(train_df)}  ->  {TRAIN_CSV}")
    print(f"测试集行数 (20%)  : {len(test_df)}  ->  {TEST_CSV}")
    print("\n训练集标签分布：")
    print(train_df["label"].value_counts().sort_index().to_string())
    print("\n测试集标签分布：")
    print(test_df["label"].value_counts().sort_index().to_string())
