# 公司类型文本分类

为投资公司构建的中文公司业务描述分类器：输入一段公司业务说明，输出其所属的公司类型（共 11 类），作为投资评估流程的第一步过滤器。

## 数据集

- 路径：`NLP-文本分类/training.csv`
- 规模：4,774 条人工标注样本
- 列结构：`label`（标签 1–11，无表头）、`text`（中文业务描述）
- 文本长度：平均约 726 字
- 类别分布严重不均衡：
  - 多数类 label 3、4 各约占 26%
  - 少数类 label 1 (54 条 / 1.1%)、label 11 (96 条 / 2.0%)

## 方法：TF-IDF + 线性分类器

经典 ML 流水线，已稳定通过 ≥80% 的交付门槛：

```
原始中文文本
  → jieba 中文分词（精确模式，过滤标点和单字符非汉字）
  → TfidfVectorizer（1-2gram，max_features=50000，min_df=2，sublinear_tf）
  → 分类器（LogisticRegression / LinearSVC）
```

针对类别不均衡，分类器使用 `class_weight="balanced"` 加权。

## 数据切分与验证策略

- **外层留出**：先按 `random_state=42` 做 80/20 分层切分，20% 测试集在建模过程中始终不接触
- **内层交叉验证**：在 80% 训练集上跑 `StratifiedKFold(n_splits=5)`，报告平均准确率 ± 标准差
- **最终评估**：用 CV 选出的最佳模型在完整 80% 训练集上重训，再到 20% 留出测试集上一次性评估

## 实验结果

5 折交叉验证（80% 训练集，3,819 行）：

| 模型 | 平均准确率 ± 标准差 | 是否达标 (≥0.80) |
|---|---|---|
| LogisticRegression | 0.8481 ± 0.0185 | ✅ 通过 |
| **LinearSVC** | **0.8686 ± 0.0173** | ✅ **通过（最佳）** |

留出测试集（955 行，使用 LinearSVC）：

- 准确率：**0.8702**
- 宏 F1：0.8609 ｜ 加权 F1：0.8693
- 所有 11 个类别召回率均 ≥ 0.50（最低为 label 1，0.6364）
- CV 准确率 (0.8686) ≈ 留出准确率 (0.8702)，无过拟合

## 项目结构

```
text_classfication/
├── README.md                       # 本文件
├── requirements.txt                # Python 依赖
├── NLP-文本分类/training.csv       # 原始数据
├── data/                           # 80/20 切分产物 (gitignore)
├── src/
│   ├── data.py                     # 加载、80/20 切分、KFold 工厂
│   ├── tokenize_zh.py              # jieba 中文分词包装
│   ├── train_classical.py          # TF-IDF + LR/SVC 5 折 CV
│   └── evaluate.py                 # 留出测试集最终评估
└── artifacts/classical/            # 模型与报告产物 (gitignore)
```

## 快速开始

```bash
# 1. 环境
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 数据切分（生成 data/train.csv 和 data/test.csv）
python -m src.data

# 3. 5 折交叉验证（交付门槛证据）
python -m src.train_classical

# 4. 留出测试集最终评估 + 序列化模型
python -m src.evaluate
```

## 推理示例

```python
import joblib
pipe = joblib.load("artifacts/classical/linear_svc_pipeline.joblib")
desc = "公司主营业务为向中小微企业提供贷款服务..."
print(pipe.predict([desc]))   # -> array([2])  标签范围 1-11
```
