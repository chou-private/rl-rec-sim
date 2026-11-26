# 有限历史下的问卷画像与 Yahoo 五分类 DeepFM＋EM

比较只看到用户 0、1、3、5 条历史主动评分时，问卷能否改善对随机歌曲评分的预测。为本实验重新训练五分类 DeepFM。输入是原始 Yahoo 文件，无需旧模型或服务器上早期实验的输出。

## 服务器运行

在项目根目录，激活装有 Python 3.10+、NumPy 和 PyTorch 2.x 的环境，并确认原始文件在 `data/YahooR3/data_raw`。如果在别处，给下列脚本追加 `--data /path/to/data_raw`。

```bash
git fetch origin
git switch codex/yahoo-sparse-history-em
git pull --ff-only origin codex/yahoo-sparse-history-em
python -m unittest discover -s tests -p 'test_yahoo_sparse*.py' -v
bash scripts/run_yahoo_sparse_em.sh smoke
```

GPU smoke 使用每个问卷组 10 人，共 50 人、全部 1,000 首歌曲、一个种子、四个历史条数、三种问卷条件。看到 `Completed. Results:` 后，在已有 tmux 会话运行正式实验：

```bash
bash scripts/run_yahoo_sparse_em.sh formal > yahoo_sparse_em_formal.log 2>&1
```

另开终端查看 `tail -f yahoo_sparse_em_formal.log`。正式设置为 3 个网络种子 × 4 种历史条数 × 3 个条件，共 36 次拟合。每次先预训练 2 轮，再运行最多 20 轮广义 EM。脚本使用 CUDA；CPU 检查可追加 `--device cpu`。默认每批 16 个用户，显存不足时所有组统一追加 `--batch-users 8` 并记录配置。输出目录须是新目录；一次运行不支持断点续训。需要分开调度时，可给每个种子指定独立的 `--seeds` 和 `--output`；各目录只汇总当次运行的种子。

## 三个条件的唯一区别

| 条件 | 五维初始画像 |
|---|---|
| global | 校准用户的平均主动评分分布，每个用户权重相同 |
| own | 用 Q2–Q6 预测本人的主动评分分布 |
| shuffled | 在用户间打乱完整五题答案，按同样方式重新拟合并预测 |

小模型是带 L2 的五类线性 softmax，以每人五档主动评分所占比例为训练目标。三组用同样的规则更新画像：`(a × 初始画像 + 可见历史的五档计数) / (a + k)`。强度 a 只用相应拟合用户的主动评分估计，各组共用。五维特征描述已记录评分的星级构成。EM 使用另估的全局五档 B 记录概率，三组和所有 k 相同。

DeepFM 接收可见的歌曲与对应星级，以及更新后的五维画像。它对每首候选歌曲输出五个星级概率；用户表示来自这些可见信息，没有用户 ID 嵌入。三组逐个训练，网络形状、同种子初始参数、批次次序、EM 参数与训练预算一致。此入口只做用户模型实验。

## 划分和遮挡

- 正式实验沿用原始分层划分（种子 20261123），3,238 校准、1,078 验证、1,084 测试用户。smoke 在按组抽取的小样本上重做分层划分，只检查流程。
- 只用校准用户拟合 DeepFM、五档 B 以及画像模型。校准用户自身的画像由另外四折用户拟合；验证、测试画像用全部校准用户拟合。整份问卷的固定打乱限制在校准折、验证池和测试池各自内部。区间没有覆盖打乱排列的不确定性。
- 每个校准用户的 1,000 首歌曲划成两个互斥的 500 首集合：按有无历史记录分层随机分半，奇数个已记录项随机向上或向下取整。一个集合提供最多 5 条可见历史；另一个作为 EM 训练目标，包含真实记录与真实缺失。前一集合的所有歌曲均排除在损失之外。这样 k 改变时，训练目标、观测比例和三组可见歌曲均可比；校准用户的其余历史只作为训练数据，验证和测试用户的隐藏历史不进入学习。
- 验证和测试的可见历史从本人的主动记录中抽样。所有条件和网络种子共用同一个历史抽样种子，较小 k 使用较大 k 的前缀。该版本原始数据的历史记录与随机评分没有用户－歌曲对重叠，运行时校验。此处是已参与问卷用户的历史遮挡，数据没有新注册用户的时间序列。
- E 步对真实缺失项使用 `q(r) ∝ p_old(r | context,item) × (1-mu[r])`，已记录目标使用真实星级。M 步用冻结的 q，一遍更新后校验完整负 Q 和观测负对数似然（均含 L2）；若有上升则降低学习率重试。验证随机歌曲的 MAE 选轮次，预训练轮 0 也可被选中。所有条件选好以后，重载各自的检查点评价测试池。

## 读取结果

正式结果位于 `visual_results/yahoo_sparse_em/formal_v1/`。优先看 k=1 时 own 相对 global 和 shuffled 的配对随机评分 MAE 差异（负值较好）；k=0、3、5 描述不同历史条数下的结果。RMSE、星级 NLL 和每人十首随机歌曲内的 NDCG@5 作为辅助指标。NDCG 以 4–5 星为相关，只平均有相关歌曲的用户。伪真值和模拟策略收益不参与选模或评价。

- `summary.json` 汇总各历史条数的跨种子均值和标准差，并标记是否 smoke。`results.json` 保留各次拟合指标。`paired_bootstrap.json` 是每个种子内按用户配对的区间，不涵盖重新拟合或重抽历史带来的变化。
- `seed_*/k_*/*/` 保存 warmup、每轮目标和验证 MAE、所选轮次、检查点和逐用户测试指标。`config.json`、`splits.json`、`recording_B.json`、`profile_fits.json` 保存配置、数据与代码哈希、用户划分、B 和画像拟合审计。本地保留 `contexts.npz` 与 `calibration_targets.npz` 可核对实际输入和监督掩码。
- 传回汇总结果供分析时，使用下面的精简包。数据、含个人画像的文件及模型权重保留在服务器。

```bash
tar -czf yahoo_sparse_em_summary.tar.gz \
  visual_results/yahoo_sparse_em/formal_v1/summary.json \
  visual_results/yahoo_sparse_em/formal_v1/results.json \
  visual_results/yahoo_sparse_em/formal_v1/paired_bootstrap.json \
  visual_results/yahoo_sparse_em/formal_v1/config.json \
  visual_results/yahoo_sparse_em/formal_v1/recording_B.json \
  yahoo_sparse_em_formal.log
```

## 本地检查

在 PyTorch 2.9.0、NumPy 2.3.4 的 CPU 环境，50 名真实用户、四个 k、三个条件、两个网络种子完成了 24 次最小拟合，48 次 EM 更新均通过两个目标的验收。另用正式 5,400 人数据检查可见历史、折外画像和训练目标的构造（共 1,619,000 个校准目标）。这些是实现检查；正式效果以服务器实验为准。
