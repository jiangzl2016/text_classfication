"""TF-IDF + 支持向量机表现更好（准确率 0.8686 ± 0.0173。在training set上重新训练该模型，然后在test set上评估"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC

from src.data import load_split
from src.train_classical import build_pipeline

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "classical"
MODEL_PATH = ARTIFACT_DIR / "linear_svc_pipeline.joblib"
REPORT_PATH = ARTIFACT_DIR / "test_report.json"
CM_PATH = ARTIFACT_DIR / "confusion_matrix.png"


def main() -> None:
    train_df, test_df = load_split()
    X_train = train_df["text"].tolist()
    y_train = train_df["label"].to_numpy()
    X_test = test_df["text"].tolist()
    y_test = test_df["label"].to_numpy()

    print(f"训练集行数：{len(X_train)}    测试集行数：{len(X_test)}")
    print("在完整 80% 训练集上重新训练 LinearSVC...")
    pipe = build_pipeline(LinearSVC(class_weight="balanced", C=1.0))
    pipe.fit(X_train, y_train)

    print("在 20% 留出测试集上进行预测...")
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0,
    )
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cls_report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 70)
    print("留出测试集评估结果")
    print("-" * 70)
    print(f"  准确率          : {acc:.4f}")
    print(f"  宏精确率        : {macro_p:.4f}")
    print(f"  宏召回率        : {macro_r:.4f}")
    print(f"  宏 F1           : {macro_f1:.4f}")
    print(f"  加权 F1         : {weighted_f1:.4f}")
    print("\n各类别详细报告：")
    print(cls_report)

    # 各类别召回率检查
    per_recall = {}
    labels_sorted = sorted(np.unique(y_test).tolist())
    for cls in labels_sorted:
        mask = y_test == cls
        per_recall[int(cls)] = float((y_pred[mask] == cls).mean())
    print("各类别召回率（少数类期望 ≥ 0.50）：")
    for cls, r in per_recall.items():
        flag = "合格" if r >= 0.50 else "偏低"
        print(f"  [{flag}] 标签 {cls:>2d}  召回率={r:.4f}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存产物文件
    joblib.dump(pipe, MODEL_PATH)
    print(f"\n已保存模型管道 -> {MODEL_PATH}")

    report = {
        "model": "LinearSVC + TF-IDF (1-2gram)",
        "test_accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_recall": per_recall,
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist(),
        "labels": labels_sorted,
    }
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"已保存测试报告 -> {REPORT_PATH}")

    # 绘制混淆矩阵
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_sorted, yticklabels=labels_sorted, cbar=True)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix — LinearSVC (test acc={acc:.4f})")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    print(f"已保存混淆矩阵 -> {CM_PATH}")

    print("\n" + "=" * 70)
    cv_gate = "通过" if acc >= 0.80 else "未通过"
    print(f"[{cv_gate}] 留出测试集准确率 {acc:.4f}，对比交付门槛 0.80")


if __name__ == "__main__":
    main()
