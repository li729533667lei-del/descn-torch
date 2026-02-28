# DESCN Torch (PyTorch 2.5.1)

一个基于 **PyTorch 2.5.1** 的可运行示例项目，实现了 **DESCN: Deep Entire Space Cross Networks** 的多头建模。

## 特性

- 支持 4 类输入特征：
  - 单值连续特征（dense scalar）
  - 序列连续特征（dense sequence）
  - 单值类别特征（sparse scalar）
  - 序列类别特征（sparse sequence）
- 支持 4 个输出头：
  - `control`
  - `treatment_1`
  - `treatment_2`
  - `treatment_3`
- 内置 demo 数据生成，可直接训练与推理。

## 环境

- Python 3.10+
- PyTorch 2.5.1

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
python train_demo.py
```

运行后将：

1. 自动生成 demo 数据集
2. 训练 DESCN 模型
3. 输出每个 epoch 的训练损失
4. 打印四个头的预测结果样例

## 目录结构

```text
.
├── descn_torch
│   ├── __init__.py
│   ├── data.py
│   └── model.py
├── requirements.txt
├── train_demo.py
└── README.md
```
