"""TF-IDF + 逻辑回归和支持向量机。用5折交叉验证来评估准确性."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data import get_kfold, load_split
from src.tokenize_zh import tokenize

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "classical"


def build_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            max_features=50000,
            min_df=2,
            sublinear_tf=True,
        )),
        ("clf", clf),
    ])


CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs",
    ),
    "LinearSVC": LinearSVC(class_weight="balanced", C=1.0),
}


def main() -> None:
    train_df, _ = load_split()
    X = train_df["text"].tolist()
    y = train_df["label"].to_numpy()
    skf = get_kfold()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name, clf in CLASSIFIERS.items():
        pipe = build_pipeline(clf)
        t0 = time.time()
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=1)
        oof_pred = cross_val_predict(pipe, X, y, cv=skf, n_jobs=1)
        elapsed = time.time() - t0

        mean, std = scores.mean(), scores.std()
        report = classification_report(y, oof_pred, digits=4, zero_division=0)
        cm = confusion_matrix(y, oof_pred).tolist()

        print("=" * 70)
        print(f"{name}")
        print("-" * 70)
        print(f"  各折准确率      : {scores.tolist()}")
        print(f"  平均 ± 标准差   : {mean:.4f} ± {std:.4f}")
        print(f"  耗时            : {elapsed:.1f}s")
        print(f"  OOF 分类报告：")
        print(report)

        summary[name] = {
            "fold_scores": scores.tolist(),
            "mean_accuracy": float(mean),
            "std_accuracy": float(std),
            "elapsed_sec": elapsed,
            "classification_report": report,
            "confusion_matrix": cm,
        }

    out_path = ARTIFACT_DIR / "cv_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("=" * 70)
    print(f"\n汇总结果已写入 {out_path}")

    print("\n5 折交叉验证最终准确率：")
    for name, info in summary.items():
        gate = "通过" if info["mean_accuracy"] >= 0.80 else "未通过"
        print(f"  [{gate}] {name:22s} {info['mean_accuracy']:.4f} ± {info['std_accuracy']:.4f}")


if __name__ == "__main__":
    main()
