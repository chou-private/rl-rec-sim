# Yahoo DeepFM + EM：仅 B/C 实验

## 范围

这是独立的新实验入口，不改变旧 DeepFM、负例权重或 DQN 的行为。
从 `codex/yahoo-survey-calibrated-reward` 的 `c90d97fe` 新建分支；只新增实验、测试和说明。
不宣称完整复现 UAI 2007 论文：保留其 rating-dependent missingness 思路，用 categorical DeepFM
代替 multinomial mixture 偏好模型，通过固定记录概率的广义 EM 训练。

- B：所有人共用 5 个记录概率。
- C：Q2—Q6 固定回答模式决定记录概率表，组内概率向全局收缩。
- 两组输入均只有 user_id、item_id；不将问卷作为偏好网络输入。
- 没有 IPS 权重、零分负例、旧 exposure/variance 损失、ensemble 或 DQN。
- 只使用有问卷的 5400 人和全部 1000 首歌。不是冷启动实验。

## 环境与启动

从仓库根目录执行。仅需 Python >=3.10、NumPy 和 PyTorch >=2.0；不依赖旧的
DeepCTR、tianshou、gym、pandas 或 sklearn。开发检查环境为 Python 3.11、PyTorch 2.9.0、NumPy 2.3.4。
服务器按已有 CUDA 驱动安装兼容的 PyTorch；CPU 也能运行。

```bash
python -c 'import torch,numpy; print(torch.__version__,numpy.__version__,torch.cuda.is_available())'
python -m unittest discover -s tests -p test_yahoo_em.py -v
```

数据保持本地，不上传 GitHub。`--data` 指向包含以下原始文件的目录：

- `ydata-ymusic-rating-study-v1_0-train.txt`
- `ydata-ymusic-rating-study-v1_0-test.txt`
- `ydata-ymusic-rating-study-v1_0-survey-answers.txt`

服务器先跑通小规模流程（每组 10 人、全部 1000 首歌，**不能作为论文性能结果**）：

```bash
python -m experiments.yahoo_em.run \
  --data data/YahooR3/data_raw \
  --output visual_results/yahoo_em/server_smoke \
  --smoke-users-per-group 10 --seeds 2022 2023 \
  --warmup-epochs 1 --em-rounds 2 --dim 8 --hidden 16 \
  --batch-size 2048 --threads 2 --device cuda
```

无 GPU 时使用 `--device cpu`。输出目录必须不存在，避免覆盖已完成实验。

正式实验（建议在 tmux 中运行）：

```bash
python -u -m experiments.yahoo_em.run \
  --data data/YahooR3/data_raw \
  --output visual_results/yahoo_em/formal_bc_v1 \
  --device cuda --seeds 2022 2023 2024 \
  --strengths 0.25 0.5 --split-seed 20261123 \
  --warmup-epochs 2 --em-rounds 5 --m-epochs 1 \
  --dim 16 --hidden 64 --batch-size 4096 --threads 4
```

首个 seed 训练 B 和 C 的 0.25、0.5 两个候选，加上 lambda=0（直接复用 B），
仅按验证 MAE 选出 C 的收缩强度，随后两个 seed 固定这个强度。每个 seed 的 B/C
共享预训练初始化和数据顺序种子，各自按验证 MAE 选择 EM 检查点。
不是给每个 seed 根据测试表现重新选择参数。若选择 0，则 C 确实等于 B，不另造改善。
本入口不提供断点恢复；中断后用新的输出目录完整重跑，保留旧目录以供审计。

## 模型与训练细节

五类 logits = 类别线性项 + 类别 FM 项 + 深层项。用户/歌曲 embedding 在 FM 与 DNN 中共享。
两字段 FM 的交互向量为逐元素乘积，用 `Linear(dim, 5, bias=False)` 映射到五个类别。
不能把同一个 FM 标量广播到五类，它会被 softmax 抵消。网络输出归一化的 1—5 星分布。
预测分数取期望 `sum(v * p_v)`；MAE 不使用后验软标签，也不对结果额外校准。

1. 用主动评分交叉熵预训练，B/C 使用完全相同的参数副本。不是新增 A 对照组。
2. E 步用上一轮网络：缺失项 `q_v ∝ p_old(v|u,i) * (1-mu_gv)`；已记录项为真实星级 one-hot。
3. M 步用冻结的 q 训练。目标是整个用户×歌曲集合的平均交叉熵，加 `l2 * sum(parameters^2)`。
   不把已记录和缺失两部分分别取平均后等权相加。无额外采样权重。
4. 每轮重新建立 Adam，顺序由 seed 和轮次决定。默认一轮 M 步遍历全部配对一次。
5. 比较完整的固定软标签目标和观测数据负对数似然（均含相同正则）。若没有下降，
   回退到该轮旧参数，学习率减半重试，最多 5 次。仍失败则回退并停止该次训练。
   一个 EM 更新都未接受会报错，不能把预训练模型冒充 EM 结果。

记录概率固定时，已记录项的 log(mu) 和缺失项的期望 log(1-mu) 对 M 步参数是常数，
因此训练损失可省略；计算观测数据似然时完整保留记录机制。
广义 EM 以完整目标验收更新；不承诺每个 minibatch 都单调，也不承诺验证 MAE 单调。
观察到的训练似然包含不同 B/C 的 mu，不能将其直接作为问卷价值的主要检验。

完整 E 步 targets 是 float32 的 540万×5 数组，约 108 MB，放 CPU；网络分批放 GPU。
每轮还会完整遍历数据计算诊断，耗时以服务器 smoke 和首轮日志为准。
没有改成缺失采样，也没有对 0—4 类索引和 1—5 星级混用。

## 数据隔离与估计

按固定问卷组分层、约 60/20/20 划分校准/验证/测试用户（分组取整，确切数量见 splits.json）。
各部分用户的历史主动评分都用于偏好模型训练。随机评分：校准部分只估计 mu，验证部分选
lambda/轮次，测试部分只在选定 lambda 后评估。随机评分标签不进入 warmup、E 或 M 步。
历史缺失掩码只由历史主动评分表生成，与随机评分星级无关。

问卷分组优先级与旧分析一致：flat → 两端均比中立高至少 1 → 正向均值较高 → 负向均值较高 → 其他。
全量人数必须为 2000/2201/934/184/81。原始 ID 从 1 开始，内部从 0 开始。

mu_v = active_count_v / (校准人数 × 1000 × 随机评分星级边际概率)。
随机边际用每星 0.5 的对称 Dirichlet 伪计数平滑；估计的 mu 限制到 [1e-5,1-1e-5]。
保存计数、原始比值、平滑边际和每个裁剪事件。组内同样处理，再使用
mu_gv = (1-lambda) * mu_v + lambda * group_mu_gv。正式解读前检查裁剪是否频繁，不能忽略异常。

这些是选择机制假设下的估计，不是实际曝光后反馈概率。历史与随机评分的时间和收集方式不同，
数据人群还经过至少十条历史评分的筛选。程序记录两表配对重合数；零重合本身不能证明未经过
发布处理或删失。该限制和“历史未评分不代表不喜欢”应在论文中保留。
整套数据过去已经用于探索，本次是内部留出，不宣称全新外部验证。

## 输出和判断标准

- `config.json`：完整参数、Git commit/dirty 状态、源代码和输入数据 SHA256、环境版本。
- `splits.json`：原始 1-based 用户 ID；`original_user_ids.npy`：矩阵行映射。
- `propensities.json`：全局/组内概率估计审计；`selected_strength.json`：验证集选择。
- 每个 seed/候选的 `history.json`：Q 和观测目标、是否接受、重试次数、验证指标、耗时。
- `best.pt`：模型 state_dict、结构参数、最佳轮次、mu；`selection.json`：无张量版元数据。
- `expected_ratings.npy`：5400×1000 期望评分（smoke 是子集行），按 original_user_ids.npy 定位。
  **不是旧 15400 行矩阵的直接替换文件**，本轮尚未接入 DQN，不能直接覆盖旧模型产物。
- `seed_*/B_test.json`、`C_test.json` 和 `*_users.csv`：总体、问卷组、低历史评分数用户诊断。
- `test_results.json`、`summary.json`：三次配对 seed 的 MAE/RMSE/NLL/NDCG 和 C−B 差值。
- `paired_user_bootstrap.json`：每个 seed 内 2000 次配对用户 bootstrap。它不涵盖训练或概率
  校准不确定性；不能把 seed×用户当独立样本来夸大显著性。

主要指标为随机测试评分 MAE。NLL 为随机评分本身的类别概率损失，不是缺失机制训练损失。
NDCG@5 在每人十首随机已评分歌曲中计算，rating>=4 为二元相关，仅统计有正例用户，
并报告人数；不是全量歌曲检索效果。排序并列时按歌曲 ID 打破并列。
优先观察 C−B 是否跨种子一致、置信区间和效应大小，而非仅挑选某个 seed 的最好结果。
lambda=0 或结果混合时，结论应为本配置下未发现稳定额外价值。

## 本地验证

见 `tests/test_yahoo_em.py`：手算后验与似然、相同记录概率的后验不变性、分组与划分、
校准不读取留出标签、FM 分支梯度、E 步冻结标签、微型 EM 目标下降和 C(lambda=0) 与 B 一致。
真实 Yahoo 子集 smoke 同时覆盖数据读取、概率估计、候选选择、训练、导出和 bootstrap。
本地只验证 CPU 执行；CUDA 正式运行需先在服务器执行上述 smoke。

2026-11-23 本地检查记录：7 项单元/微型 EM 检查通过；50 人和 100 人真实数据子集、
各两个 seed 的端到端检查通过。分别覆盖 lambda=0 回退与非零 lambda 后续 seed 路径。
100 人检查的 5 次模型训练共 10 轮 EM 全部接受更新，完整目标均未上升，导出的
100×1000 评分矩阵均有限且落在 [1,5]。这些小规模检查不作为效果证据。
全量只做读取和概率估计预检，没有本地运行正式训练：分层划分为 3238/1078/1084 人，
所有组的记录概率估计均未触发裁剪。正式服务器运行仍以日志审计为准。
