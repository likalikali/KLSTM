# 原始脚本与论文图表映射

这份映射的目的，是告诉你“原仓库里哪些脚本真的对论文有用”，以及“整理后的 `repro/` 入口如何把它们替换成可重复的重算流程”。

| 论文对象 | 当前推荐入口 | 原始来源 | 当前状态 |
|---------|-------------|---------|---------|
| Figure 7 | `repro/scripts/generate_figure7.py` | `数据展示.py` + `dataset2.csv` | 已改成原始数据直接重算 |
| Table 1 | `repro/scripts/generate_figure7.py` | 论文描述 + `dataset2.csv` | 已改成原始数据直接统计 |
| Figure 8 | `repro/scripts/generate_figure8.py` | `LSTM对比实验.py` + `dataset2.csv` + `xxx.csv`/`zzz.csv`/`zzzhat.csv` 线索 | 已改成训练生成三线图 |
| Figure 8 历史快照 | `repro/scripts/export_archived_figure8.py` | `dataAnalysis/modelFit.py` | 仅保留历史 FM-GRU 双线对照，输出名已改为 `historical_reference_figure8_two_line.*` |
| Table 2 | `repro/scripts/generate_table2.py` | `柱状图.py` + 多个旧模型脚本 | 已改成统一训练/评估重算 |
| Table 3 | `repro/scripts/generate_table3.py` | 论文正文 + FM 模块消融思路 | 已改成真实消融实验 |
| Figure 9 | `repro/scripts/generate_figure9.py` | `k取值作图.py` + `k的取值的影响*.csv` | 已改成真实 K sweep |
| Figure 10 | `repro/scripts/generate_figure10.py` | `3d图.py` | 已改成真实 batch_size × learning_rate sweep |
| Table 4 | `repro/scripts/generate_table4.py` | 论文正文 + 旧 PH 泛化实验线索 | 已改成真实 PH 目标泛化实验 |

## 关于训练主线

### 不推荐继续直接使用的入口
- `main.py`
- `main1.py`
- 根目录 `model.py`

原因：
- 根目录 `model.py` 更像 FC-LSTM/FC-GRU 旧对比模型，不是当前最接近论文的 FM-GRU 主线。
- `main1.py` 和图表、测试、实验代码混在一起，不适合作为复现入口。

### 当前整理后的训练入口
- `repro/scripts/run_fm_gru.py`
- `repro/scripts/generate_figure8.py`

对应关系：
- `run_fm_gru.py`：清理后的 FM-GRU 主线训练
- `generate_figure8.py`：训练 FM-GRU + 训练 FC-LSTM，并导出论文 Figure 8 的三线图

## 关于 Figure 8 的关键线索

目前训练版 Figure 8 用到的关键线索如下：

1. `dataset2.csv` 提供 Figure 8 对应的 60 个时间步原始观测值
2. `trainindex.csv` 提供 FM-GRU 固定训练索引
3. `LSTM对比实验.py` 提供 FC-LSTM 的旧训练口径
4. `xxx.csv` / `zzz.csv` / `zzzhat.csv` 只作为反查和校验线索，不再作为最终输出数据源

## 关于一键复现入口

- `repro/scripts/run_paper_repro.py` 默认只运行“原始数据重算 + 训练版 Figure 8”。
- 需要时可以显式传入 `--include-tables --include-figure9 --include-figure10`，把整篇论文的图表都切到重算链路。
- 历史双线 Figure 8 快照只有在显式传入 `--include-archived-reference` 时才会导出。
