# Battery Cathode AI

锂离子电池正极材料性能预测工具包：基于 CGCNN 的多任务学习模型，同时预测放电比容量、工作电压和循环寿命，并通过 Monte Carlo Dropout 给出不确定度。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
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
  data_loader.py      #数据加载
  graph_builder.py    # CIF -> 晶体图（当前为 mock 实现）
  loss.py             #多任务损失
  trainer.py          #训练循环
  predict.py          #预测 + MC Dropout UQ
  models/
    cgcnn_mtl.py      # CGCNN-多任务模型
config.yaml
online-json-editor.json   # 基准材料性能
```
