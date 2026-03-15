# FM-GRU 复现整理说明

这个目录是对原始工程的“干净复现层”。老师原始脚本和历史实验文件仍保留在项目根目录，但论文图表和表格的推荐入口已经统一迁移到这里。

## 目录结构

```text
repro/
├── src/                  # 清理后的数据处理与模型模块
├── scripts/              # 训练脚本、作图脚本、一键复现脚本
├── output/
│   ├── figures/
│   ├── tables/
│   └── runs/
└── legacy_map.md         # 原始脚本与论文图表的对应关系
```

## 当前复现分层

### 1. 原始数据直接重算
- Figure 7
- Table 1

### 2. 从头训练或评估生成
- Figure 8
- Table 2
- Table 3
- Table 4
- Figure 9
- Figure 10
- FM-GRU 主线训练指标与拟合图

### 3. 历史对照
- `export_archived_figure8.py`

说明：
- `export_archived_figure8.py` 只是历史 FM-GRU 双线快照，不是论文主 Figure 8。
- 当前主链路已经不再依赖论文 Markdown、旧 CSV 网格或硬编码数组来生成 Table 2 / 3 / 4、Figure 9 / 10。

## 环境

默认使用项目根目录下的 `.venv`：

```powershell
& ".\.venv\Scripts\python.exe" --version
```

至少需要以下依赖：

```text
torch
numpy
pandas
scikit-learn
matplotlib
tqdm
openpyxl
xgboost
statsmodels
```

## 运行命令

### 1. 生成 Figure 7 和 Table 1

```powershell
& ".\.venv\Scripts\python.exe" "repro\scripts\generate_figure7.py"
```

输出：
- `repro/output/figures/figure7_covariates.png`
- `repro/output/tables/table1_statistics.csv`

### 2. 训练生成 Figure 8

```powershell
& ".\.venv\Scripts\python.exe" "repro\scripts\generate_figure8.py" --device cuda --epochs 10
```

输出：
- `repro/output/figures/figure8_trained_comparison.png`
- `repro/output/tables/figure8_trained_comparison.csv`
- `repro/output/tables/figure8_trained_summary.json`

说明：
- 默认会训练 FM-GRU 和 FC-LSTM，且默认 `batch_size=4`，更贴近论文设置。
- Figure 8 使用和论文图面一致的 60 个时间步块做对比。
- FC-LSTM 走的是更贴近原始 `LSTM对比实验.py` 的旧链路训练口径。
- 如果根目录存在 `xxx.csv`，会优先用它作为旧 Figure 8 的 60 步输入窗口，以减少与论文原图的偏差。

### 3. 单独训练 FM-GRU 主线

快速验证：

```powershell
& ".\.venv\Scripts\python.exe" "repro\scripts\run_fm_gru.py" --epochs 1 --batch-size 4 --lr 0.001 --device cuda
```

按固定索引切分跑一版：

```powershell
& ".\.venv\Scripts\python.exe" "repro\scripts\run_fm_gru.py" --data-path dataset2.csv --train-index-path trainindex.csv --test-index-path testindex.csv --epochs 10 --batch-size 4 --lr 0.001 --hidden-size 128 --num-layers 2 --encode-step 24 --forecast-step 12 --fm-k 84 --device cuda
```

输出：
- `repro/output/runs/fm_gru_clean/metrics.csv`
- `repro/output/runs/fm_gru_clean/metrics.json`
- `repro/output/runs/fm_gru_clean/train_loss.csv`
- `repro/output/runs/fm_gru_clean/last_batch_predictions.csv`
- `repro/output/runs/fm_gru_clean/figure8_like_last_batch.png`

### 4. 生成 Table 2 / Table 3 / Table 4

```powershell
& ".\.venv\Scripts\python.exe" "repro\scripts\generate_table2.py" --device cuda --epochs 10 --repeats 10
& ".\.venv\Scripts\python.exe" "repro\scripts\generate_table3.py" --device cuda --epochs 10 --repeats 10
& ".\.venv\Scripts\python.exe" "repro\scripts\generate_table4.py" --device cuda --epochs 10 --repeats 10
```

说明：
- 这三张表都已经改成基于模型和数据集重算。
- `export_paper_tables.py` 现在只是兼容旧入口，内部会转调新的三个脚本。
- 运行过程中会显示 `tqdm` 进度条，能看到当前模型、重复轮次和训练推进情况。

### 5. 生成 Figure 9 / Figure 10

```powershell
& ".\.venv\Scripts\python.exe" "repro\scripts\generate_figure9.py" --device cuda --epochs 10 --repeats 1
& ".\.venv\Scripts\python.exe" "repro\scripts\generate_figure10.py" --device cuda --epochs 10 --repeats 1
```

说明：
- 这两个脚本已经改成真实 sweep，不再读取旧的 `k的取值的影响*.csv` 或硬编码 RMSE 网格。

### 6. 导出历史 FM-GRU 双线快照

```powershell
& ".\.venv\Scripts\python.exe" "repro\scripts\export_archived_figure8.py"
```

说明：
- 这个脚本只保留作历史对照，不再作为论文主 Figure 8 入口。
- 输出文件名已改成 `historical_reference_figure8_two_line.*`，避免和训练版 Figure 8 混淆。
- 更早一版误导性的 `figure8_archived_fit.*` 已移动到 `repro/output/legacy/` 留档。

### 7. 一键执行复现流程

默认只包含 Figure 7、Table 1 和训练版 Figure 8：

```powershell
& ".\.venv\Scripts\python.exe" "repro\scripts\run_paper_repro.py" --figure8-epochs 10 --figure8-device cuda
```

如果还想额外跑一版 FM-GRU 主线：

```powershell
& ".\.venv\Scripts\python.exe" "repro\scripts\run_paper_repro.py" --figure8-epochs 10 --figure8-device cuda --include-training --training-epochs 10 --training-device cuda
```

如果需要连同 Table 2 / 3 / 4 与 Figure 9 / 10 一起跑：

```powershell
& ".\.venv\Scripts\python.exe" "repro\scripts\run_paper_repro.py" --figure8-epochs 10 --figure8-device cuda --include-tables --include-figure9 --include-figure10 --table-repeats 10 --sweep-repeats 1
```

如果还要附带导出历史快照对照：

```powershell
& ".\.venv\Scripts\python.exe" "repro\scripts\run_paper_repro.py" --figure8-epochs 10 --figure8-device cuda --include-archived-reference
```

## 建议的复现顺序

1. 先跑 `generate_figure7.py`
2. 再跑 `generate_figure8.py`
3. 再跑 `generate_table2.py`、`generate_table3.py`
4. 再跑 `generate_figure9.py`、`generate_figure10.py`
5. 最后跑 `generate_table4.py`

这样拿到的就是“图 7 / 8 / 9 / 10 和表 1 / 2 / 3 / 4 全部重算”的完整结果。
