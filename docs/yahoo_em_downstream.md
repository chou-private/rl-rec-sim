# EM B/C → Greedy / Double DQN

独立入口：`python -m experiments.yahoo_downstream.run`。只需 NumPy / PyTorch，复用 EM 文件和
Yahoo 固定伪真值、歌曲距离数据。旧 DQN/DeepFM 文件不改动；本实现的检查点与旧 DQN 不兼容。
这是一套受控的下游实验，不是对旧 DQN 日志的严格复现。不能将模拟结果称为在线效果。

## 输入

`--em-root` 指向服务器保留的完整 50 轮 EM 输出目录。需要每个 seed 的 `B/best.pt`、
`C_selected.json` 指向的 checkpoint 及其同目录 `expected_ratings.npy`，以及根目录
`original_user_ids.npy`、`groups.npy`、`splits.json`。之前上传的轻量结果压缩包不含模型和矩阵，
不能作为此输入。自动解析实际选中的 C，不硬编码 lambda。用户 ID 按映射读取，不假设子集连续。

`--yahoo-root` 默认为 `data/YahooR3`，需要：

- `data_raw/RL4Rec_data/yahoo_pseudoGT_ratingM.ascii`
- `data_processed/distance_mat.pickle`（可信本地文件）
- `data_raw/ydata-ymusic-rating-study-v1_0-train.txt`（只用于评分数量分组）

## 固定设计

- 四组合：B/C × Greedy/DQN。Greedy 按期望评分排序，无训练。
- DQN 训练用户池用 calibration，验证/测试沿用原划分。训练使用所有校准用户，逐次随机抽样；
  验证和测试对指定用户逐一评估一次。原 EM 已见过各池用户的历史主动评分，不是纯冷启动。
- 每个 seed 的两组共用**B 检查点的冻结用户及歌曲嵌入**。状态为用户向量拼接最近三步
  `(歌曲向量, 反馈)` 的均值；空历史是零向量。用户向量使首次推荐能区分用户。
  与旧 tracker 默认不含用户嵌入不同，两组同步采用。嵌入不进入优化器。
- 标准 Double DQN：128/128 ReLU 网络，Adam lr=.001，gamma=.9，单步 TD，Huber loss，
  梯度范数上限 10，目标网络每 320 次梯度更新复制，epsilon=.1，均匀回放容量 20000，
  batch 128，前 1000 次环境交互后学习，每交互一次更新一次。默认 10 epoch × 10000 步。
  是独立 PyTorch 实现，不导入旧 tianshou 的状态跟踪/collector 耦合。
- B/C 同一 seed 的网络初始化一致；随机流种子配对。由于策略/退出不同，实际轨迹不保证相同。
- 训练反馈固定为 `(预测评分-1)/4`，不按矩阵最小值平移，无额外问卷奖励或退出惩罚。
- 评估只用共同 pseudo-GT；反馈进状态前也做 `(评分-1)/4`，报告 reward 为原始 1—5 分。
- 所有场景禁止会话内重复；历史主动评分歌曲不额外排除，四组合一致。
- 自然退出沿用距离小于120的规则，窗口3、最多30步，计入触发退出的动作。
  修正历史不足3步时负起点切片丢失历史的问题：始终取最近最多3步。该修正统一应用四组合。
- 固定10步评价忽略自然退出，不重复。DQN 训练使用自然退出。
- **只按验证集固定10步平均累计 reward 选择 DQN checkpoint**，并列取最早；自然退出指标
  不另挑模型。选完后加载保存的 checkpoint 再测。测试不参与更新或选择。
- 三个 EM seed 各配一个同号 DQN seed；这是整条流程的三次配对重复，不能分离两种训练随机性。

## 服务器命令

仓库切换至 `codex/yahoo-em-downstream`，激活原 EM 环境。从仓库根目录执行：

```bash
python -m unittest discover -s tests -p test_yahoo_downstream.py -v

python -u -m experiments.yahoo_downstream.run \
  --em-root visual_results/yahoo_em/formal_bc_50rounds_v1 \
  --output visual_results/yahoo_downstream/server_smoke \
  --seeds 2022 --smoke-users 5 --epochs 2 --steps-per-epoch 100 \
  --learning-starts 20 --batch-size 16 --hidden 16 --capacity 500 \
  --device cuda > yahoo_downstream_smoke.log 2>&1
```

smoke 按已有划分取少量用户，仅检查流程，不用于效果结论。正式运行在已有 tmux 会话中：

```bash
python -u -m experiments.yahoo_downstream.run \
  --em-root visual_results/yahoo_em/formal_bc_50rounds_v1 \
  --output visual_results/yahoo_downstream/formal_v1 \
  --seeds 2022 2023 2024 --epochs 10 --steps-per-epoch 10000 \
  --device cuda > yahoo_downstream_formal.log 2>&1
```

不想先训练时，可使用 `--greedy-only` 和另一个新输出目录，仅运行两组 Greedy。
输出目录必须不存在；当前不支持中断恢复。CPU 检查用 `--device cpu`。
每个 epoch 打印一次验证日志，结束显示 `Completed. Results: ...`。
默认含六次 DQN 训练，验证逐用户进行，运行时间应以服务器 smoke 和第一个 epoch 为准。

## 输出及限制

- `config.json`：配置、源码/环境数据哈希和环境版本。
- 每 seed 的 `artifact_audit.json`：矩阵、共享嵌入、映射及划分哈希；`user_pools.json`：实际用户。
- 每 B/C 的 `history.json`、`selection.json`、`best_dqn.pt`：训练诊断、验证选择和模型。
- `Greedy_fixed_test.json` 等四种方法/场景文件：逐用户动作序列、reward、长度及分组均值。
- `results.json`、`summary.json`：每 seed 的四组合，两种场景。
- `paired_bootstrap.json`：C−B（分别在Greedy/DQN）以及 DQN−Greedy（分别在B/C）的用户配对区间。
  只涵盖固定模型的用户不确定性，不包含模型估计不确定性；不能把三个seed×用户视为独立重复。
- 覆盖率、首选集中度和推荐次数集中度均为同一用户池上的统计。不使用测试表现调整预算或reward。

该实验能检验问卷改变训练反馈后的下游影响，不能单独证明 EM 优于无去偏模型，也不能证明线上
长期满意度改善。共享 B 表示是控制条件，结论不能泛化成“完整 C 表示没有价值”。

结果打包（无需原始数据或模型权重）：

```bash
tar --exclude='*.pt' -czf yahoo_downstream_results.tar.gz \
  visual_results/yahoo_downstream/formal_v1 yahoo_downstream_formal.log
```

## 本地检查记录（2026-11-24）

CPU 小规模流程使用本机已有的 EM 子集模型，不是服务器 50 轮全量模型。两个种子的 B/C
各运行2个epoch、每epoch100步；覆盖模型读取、DQN更新、保存重载、验证选模及四组合评估。
另用 lambda=0 的 EM 产物核验：B/C 的 Greedy、DQN 在两种场景中的逐用户结果完全一致。
检查相同网络初始化、无重复动作、固定10步、每个测试用户恰好一次。
单元检查覆盖状态个性化、短历史退出边界、Double DQN target动作掩码及终止处理、
回放掩码复制、共同评估奖励，以及改变测试reward不影响训练/检查点选择。
CUDA 需在服务器先执行 smoke；不把本机小样本指标作为研究结果。
