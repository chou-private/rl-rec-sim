# 固定记录网络接入 DeepFM＋EM：global / own / shuffled

入口 `python -m experiments.yahoo_recording.em`。三个记录机制来自已完成的独立记录网络实验，
全部冻结，只训练评分预测 DeepFM；不接DQN、不再估计问卷权重、不更新记录网络。

## 输入及模型对应

`--recording-root` 指向服务器上完整的 `visual_results/yahoo_recording_network/formal_v1`，需要：

- 根目录 `global.pt`、`global_selection.json`、`config.json`、`summary.json`、`data_audit.json`、
  `splits.json`、`questionnaire_mapping.json`；
- 每个seed的 `own.pt`、`shuffled.pt` 及相应 `*_selection.json`。

此前发给助手的精简压缩包排除了 `.pt`，不能作为这个输入。
不需要之前50轮EM模型；本次从共同初始化重新训练，不能直接与旧B的数值作因果归因。

- global：使用本次记录网络实验重新拟合的全局概率；
- own：原用户Q2—Q6 → 已选好的本人记录网络 → 每人五个概率；
- shuffled：使用**原记录实验保存的固定donor映射**与已选打乱网络，不重新打乱。
- 记录网络附带的星级辅助模型 p(Q) **不进入本次EM**。本次 p(v|u,i) 由 DeepFM 学习。
  所以先前改善记录统计不保证现在改善评分预测，这是本实验要验证的转移。

## 训练控制

原始文件SHA256与记录实验必须一致；用户划分必须完全匹配；donor映射必须在原池内。
正式使用原5400人，所有人的历史主动评分可以训练偏好网络；校准随机评分不额外加入监督。
验证随机评分选择EM轮次，测试随机评分在三个模型都选定后评价。
记录诊断阶段的测试用户历史记录在本阶段可以进入偏好模型训练——这是已有用户评分补全，
并非冷启动或延续上一阶段对历史记录的留出预测任务。

三个模型共享每个seed的一份预训练参数；架构、数据顺序种子、训练预算、正则一致。
E步使用每个用户五个冻结的 mu：`q_v ∝ p_old(v|u,i)*(1-mu_uv)`；
M步沿用原全配对交叉熵和正则，不加IPS权重或零分负例。网络概率只查表，不参与反向传播。
保留完整Q/观测似然验收、学习率回退和验证MAE选模。EM最大轮数默认50。
若shuffled在记录阶段退回global，两个概率矩阵及下游结果应一致；仍实际训练两组，作为一致性检查。

同一个seed对应记录网络与偏好网络的一次配对实验。global记录模型跨seed共享，
DeepFM仍有三个初始化；不把它当成三个独立数据集或三次独立记录估计。
同一数据已经多次用于研究探索，结论仍属于内部验证。概率估计和选择不确定性未包含在用户bootstrap中。

## 服务器命令

切到 `codex/yahoo-recording-em`，激活原NumPy/PyTorch环境，在项目根目录运行。

```bash
python -m unittest discover -s tests -p test_yahoo_recording_em.py -v

python -u -m experiments.yahoo_recording.em \
  --recording-root visual_results/yahoo_recording_network/formal_v1 \
  --data data/YahooR3/data_raw \
  --output visual_results/yahoo_recording_em/server_smoke \
  --seeds 2022 --smoke-users-per-split 10 \
  --warmup-epochs 1 --em-rounds 2 --dim 8 --hidden 16 \
  --batch-size 2048 --device cuda \
  > yahoo_recording_em_smoke.log 2>&1
```

smoke从原三个池各选最多10人，保留原完整donor映射，因此donor可以不在smoke子集中。
只验证接入、映射、训练、保存与评价，不用于论文性能比较。
看到 `Completed` 后正式运行（在已有tmux会话中，无需nohup）：

```bash
python -u -m experiments.yahoo_recording.em \
  --recording-root visual_results/yahoo_recording_network/formal_v1 \
  --data data/YahooR3/data_raw \
  --output visual_results/yahoo_recording_em/formal_v1 \
  --seeds 2022 2023 2024 --warmup-epochs 2 \
  --em-rounds 50 --m-epochs 1 --dim 16 --hidden 64 \
  --batch-size 4096 --lr 0.001 --l2 0.000001 \
  --device cuda > yahoo_recording_em_formal.log 2>&1
```

共9次EM训练，已完成的记录网络不会重新训练。输出目录必须不存在；不支持中断恢复。
每个EM轮次输出一次；最后 `Completed. Results: ...` 表示全部完成。
无CUDA可用 `--device cpu`，正式全量运行推荐服务器GPU。

## 输出与主要判断

- `config.json`、`data_audit.json`：源码、数据、来源配置哈希及实际样本数量。
- `splits.json`、`original_user_ids.npy`、`groups.npy`：下游实际用户及矩阵行映射。
- 每seed `recording_audit.json`：记录checkpoint哈希、冻结mu的哈希和范围、shuffled是否等于global。
- `initialization.json`：三组共用的DeepFM初始参数哈希。
- 每模型目录 `recording_mu.npy`、`history.json`、`selection.json`、`best.pt`、`expected_ratings.npy`、`test.json`。
- `results.json`、`summary.json`、`paired_bootstrap.json`：本人−全局、打乱−全局、本人−打乱。

主要比较随机评分MAE；同时看NLL、RMSE、候选NDCG@5及分组结果。
记录NLL与这里随机评分NLL不是同一个量，不能直接比较大小。
不因为某一个seed改善就宣称去偏成功；即便本人改善，也暂不自动接DQN。
当前结果目录是新的三组格式，不能直接套用旧下游仅B/C加载器。

打包给助手：

```bash
tar --exclude='*.pt' --exclude='*.npy' \
  -czf yahoo_recording_em_results.tar.gz \
  visual_results/yahoo_recording_em/formal_v1 yahoo_recording_em_formal.log
```

## 本地检查

新增检查覆盖跨池donor拒绝、条件checkpoint加载及真正使用打乱输入、global回退，
以及超过五个用户时逐用户mu索引正确、共享概率与原E步相同。
真实子集CPU短程跑两个seed、三个模型、各两轮EM，核对用户映射、记录冻结、
概率相同时训练结果一致及指标汇总。原EM回归检查保持通过；正式CUDA需先运行服务器smoke。
