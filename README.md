# Battery Cathode AI

锂离子电池正极材料性能预测工具包：基于 CGCNN 的多任务学习模型，同时预测放电比容量、工作电压和循环寿命，并通过 Monte Carlo Dropout 给出不确定度。

> **当前状态：实验性流程脚手架。** CIF 图构建目前使用确定性伪随机特征，并非真实晶体结构解析；模型和数据流程也尚未构成经过验证的科学预测系统。使用前请阅读[模型范围与当前局限](docs/MODEL_SCOPE_AND_LIMITATIONS.md)。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m battery_cathode_ai.predict
```

训练示例（默认用 mock CIF 图，不需要真实 CIF 文件）：

```bash
python -m battery_cathode_ai.trainer --config config.yaml
```

## 项目结构

```
battery_cathode_ai/
  data_loader.py      # 数据加载
  graph_builder.py    # CIF -> 晶体图（当前为 mock 实现）
  loss.py             # 多任务损失
  trainer.py          # 训练循环
  predict.py          # 预测 + MC Dropout UQ
  models/
    cgcnn_mtl.py      # CGCNN 风格多任务模型
config.yaml
online-json-editor.json   # 基准材料性能
```

## 科学使用前须知

当前代码适合验证软件流程，不适合直接报告材料性能预测结果。主要限制包括：尚未解析 CIF 结构、图卷积尚未实现完整的 CGCNN 消息传递、缺失标签默认填零、区间标签默认取均值，以及验证集划分尚未针对材料族群进行防泄漏设计。详细说明和改进清单见[模型范围与当前局限](docs/MODEL_SCOPE_AND_LIMITATIONS.md)。
