# 记录网络与 DeepFM 联合广义 EM

入口：`python -m experiments.yahoo_recording.joint`。只比较 global / own / shuffled 的随机评分预测；不接 DQN。
继承固定记录实验的完整数据哈希校验、原始划分、原 donor 映射、共同 DeepFM 初始化、评估与 bootstrap。
来源仍为完整 `visual_results/yahoo_recording_network/formal_v1`（必须含 `.pt`）；不用重新跑独立记录网络。

## 方法与固定设置

每轮先用旧的两个网络计算 E 步：已记录为真实 one-hot；缺失为
`q(u,i,r) ∝ p_old(r|u,i) * (1-mu_old(u,r))`。本轮 M 步中 q 不变化。

1. DeepFM：沿用全用户×歌曲平均软标签 CE + `l2 * sum(theta²)`，每轮一遍。
2. 记录机制：按用户和星级汇总真实记录数 `A(u,r)` 与缺失软计数 `M(u,r)`，最小化
   `-sum[A log(mu) + M log(1-mu)] / (n_users*n_items)`。
3. 加固定锚点约束：`anchor_strength * mean_{u,r} KL(Bern(mu_initial) || Bern(mu_current))`。
   锚点来自各组独立记录模型，整个运行不变。默认强度 1、记录 LR .001、每轮 5 步，三组相同。
   这是正则化设计，不是由数据估计出的“正确权重”，并不解决可辨识性问题。

global 更新五个 logits；own/shuffled 更新原 5→8→5 条件网络。条件网络原 baseline buffer 保持固定，
但末层 bias 可学习公共偏移，因此各组记录概率都可更新。问卷映射不重抽，初始独立网络也不重新拟合。
独立实验中的辅助星级模型 p(Q) 不进入联合 EM。

完整负 Q = DeepFM软CE + 记录损失 + DeepFM L2 + 锚点KL。
观测目标 = 主动评分/缺失事件的平均负对数似然 + DeepFM L2 + 同一锚点KL。
两者在同一 E 步内均不升才接受，最多五次同时减半两个学习率重试；两个网络同步恢复。
这是带正则的广义 EM，不要求无正则观测 NLL 单调，也不保证随机评分误差单调。
日志 `q_before/q_after` 是负 Q，越低越好；只能在同一轮比较，不能将不同轮的 Q 直接当成同一损失。

使用所有现有用户的历史主动记录/缺失模式更新两个网络（已有用户补全，不是冷启动）。
随机校准评分只通过初始记录模型影响此阶段，不额外作为 DeepFM 监督或记录伪标签。
验证随机评分仅按 MAE 选轮次，包含 round 0；测试随机评分在三组都选好后评价。
共同保存选中轮的两个网络，而非最佳 DeepFM 搭配最后一轮记录网络。
没有任何可接受更新则报错，不输出 completed；后续无法接受则回退并正常结束。

与之前固定实验的比较是补充：新版本也允许选 round 0，且增加记录更新及其预算。
主要公平对照仍是本次三组。打乱网络即使初始等于 global，联合更新后也不必相同，因为参数化和输入不同。
三组锚点各自来自旧模型，比较的是整个初始化+更新方案，不是仅替换问卷输入的单一变量实验。
数据集已多次探索，属于内部验证。用户 bootstrap 不含训练、记录估计及选模的不确定性。

## 服务器运行

在项目根目录，使用原 PyTorch/NumPy 环境。先拉分支：

```bash
git fetch origin
git switch codex/yahoo-recording-joint-em
git pull --ff-only origin codex/yahoo-recording-joint-em
python -m unittest discover -s tests -p 'test_yahoo_recording*.py' -v
```

先运行 GPU 小检查（新目录，不覆盖以前结果）：

```bash
python -u -m experiments.yahoo_recording.joint \
  --recording-root visual_results/yahoo_recording_network/formal_v1 \
  --output visual_results/yahoo_recording_joint/server_smoke \
  --seeds 2022 --smoke-users-per-split 10 \
  --warmup-epochs 1 --em-rounds 3 --dim 8 --hidden 16 \
  --batch-size 2048 --device cuda \
  > yahoo_recording_joint_smoke.log 2>&1
```

出现 `Completed` 后，在已有终端/tmux 会话运行正式实验，无需 nohup：

```bash
python -u -m experiments.yahoo_recording.joint \
  --recording-root visual_results/yahoo_recording_network/formal_v1 \
  --data data/YahooR3/data_raw \
  --output visual_results/yahoo_recording_joint/formal_v1 \
  --seeds 2022 2023 2024 --warmup-epochs 2 \
  --em-rounds 50 --m-epochs 1 --dim 16 --hidden 64 \
  --batch-size 4096 --lr 0.001 --l2 0.000001 \
  --recording-lr 0.001 --recording-steps 5 --anchor-strength 1 \
  --device cuda > yahoo_recording_joint_formal.log 2>&1
```

共 9 次训练。每轮打印一条 JSON，可在另一个终端 `tail -f yahoo_recording_joint_formal.log`。
输出目录必须不存在，不支持中断恢复。默认预算 50 轮，不自动增加轮数或扫约束强度。

## 输出和判读

保留原固定实验的 results/summary/paired_bootstrap、源码与数据审计、逐用户 test.json。
每组另保存：

- `initial_recording_mu.npy`：固定锚点。
- `recording_mu.npy`：最佳验证轮次的概率，与 `best.pt` 的两个网络对应。
- `best.pt`：`state_dict` 为 DeepFM，`recording_state_dict` 为记录网络。
- `history.json`：完整Q、带约束观测目标、约束值、概率范围/平均及最大漂移、验证指标、重试次数。
- `selection.json`：最佳轮次（可能为0）、接受轮数、停止原因、最佳轮次平均概率漂移。

主要看 own 相对 global 与 shuffled 的配对随机评分 MAE；同时看 RMSE/NLL/NDCG@5。
训练目标更好、记录概率发生变化、一个种子改善，都不能单独证明问卷带来预测收益。

完成后发精简包与日志；服务器保留 `.pt`、`.npy`：

```bash
tar --exclude='*.pt' --exclude='*.npy' \
  -czf yahoo_recording_joint_results.tar.gz \
  visual_results/yahoo_recording_joint/formal_v1 yahoo_recording_joint_formal.log
```

## 本地验证

4 个新增测试：汇总软计数损失/梯度与逐配对公式一致及无约束解析极值、锚点性质、
全局/条件网络联合目标下降与匹配 checkpoint、强制五次拒绝后两个网络同时回退。
另运行原 EM、固定记录接入及独立记录网络共 16 个回归测试。
真实 Yahoo 子集：三个原用户池各 10 人、两个 seed、三组各 3 轮 CPU 训练。
本地检查只验证实现，不代表正式实验有收益；CUDA 仍需服务器 smoke。
