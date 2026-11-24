# B + 问卷辅助 reward：original / own / shuffled

新入口 `python -m experiments.yahoo_downstream.questionnaire_reward`，基于已有50轮 EM **B**，
不使用 C 的预测训练，不重新训练 EM。三组重新配对训练 DQN，避免依赖旧产物缺失或配置差异。
加载器沿用完整 B/C 实验目录的审计接口，因此保留原完整 EM 目录即可。

## 预先固定的实验定义

对 Q2—Q6（原始1—5有序答案）计算 `s_uv=(q_uv-mean_v(q_uv))/4`。五题相同则全部为零。
从 B 的 `best.pt` 重算五类概率 `p_B(v|u,i)`，核对其期望与保存矩阵一致（最大误差≤1e-4）。

```
base = (B_expected_rating - 1) / 4
aux = 1[B_expected_rating >= 3] * sum_v(p_B(v|u,i) * s_uv)
original reward = base
own reward = base + 0.1 * aux(本人问卷)
shuffled reward = base + 0.1 * aux(打乱后的问卷)
```

- 默认 `alpha=0.1`、门槛3分固定，不按测试结果搜索。辅助分数是有符号相对意愿，不截断负值，
  不根据用户/模型矩阵做 min-max 缩放。每个 s 在[-0.8,0.8]，辅助改变量绝对值不超过0.08。
- 这是问卷有序答案的设计性编码，**不是已校准的反馈概率**，门槛也不是实际满意度保证。
  正向意愿可能与评分奖励重合，当前环境没有真实评分参与反馈，不能声称提高实际参与率。
- 打乱完整五题向量（不打乱题目），在原 calibration/validation/test 用户池内分别排列。
  固定 `shuffle_seed=20261124`，各模型种子使用同一映射。保存完整 donor 对应，不强制错排。
  这是一个固定随机对照，不覆盖所有可能的问卷排列；结论不应夸大其稳健性。
- smoke 先构建原池的打乱映射，再截取少量用户；smoke不用于效果推断。
- **辅助项只加入 replay 中的 TD reward**。历史状态仍接收不加分的 B 预测反馈，
  三组共用同一 B 用户/歌曲冻结嵌入。这样不把训练时的问卷加分泄露成评估时不存在的反馈。
- 验证/测试均使用原始共同 pseudo-GT reward，绝不加入问卷奖励。按验证固定10步 reward
  选择 checkpoint；测试固定10步和自然退出。用户划分、距离退出修正、动作掩码、训练预算
  沿用 `docs/yahoo_em_downstream.md`。默认三种子×三reward，共九次DQN训练。
- 不用 C 的效果为这个新方案背书，不因新结果无改善再不断修改公式。

## 服务器运行

切换 `codex/yahoo-questionnaire-reward` 分支，激活 NumPy/PyTorch 环境，从仓库根目录运行。
输入是保留 `.pt`、`.npy` 的完整50轮实验目录，不是发给助手的轻量压缩包。

```bash
python -m unittest discover -s tests -p test_yahoo_questionnaire_reward.py -v

python -u -m experiments.yahoo_downstream.questionnaire_reward \
  --em-root visual_results/yahoo_em/formal_bc_50rounds_v1 \
  --output visual_results/yahoo_questionnaire_reward/server_smoke \
  --seeds 2022 --smoke-users 5 --epochs 2 --steps-per-epoch 100 \
  --learning-starts 20 --batch-size 16 --hidden 16 --capacity 500 \
  --alpha 0.1 --rating-threshold 3 --device cuda \
  > yahoo_questionnaire_reward_smoke.log 2>&1
```

检查日志中 reward_audit：非零比例、适用门槛比例以及辅助项幅度。如果提示辅助项全为零，
不要把相同结果当作问卷无效的实证，先检查模型与分数分布；也不要依据测试效果随意降低门槛。

正式运行（已有 tmux 会话，不需要 nohup）：

```bash
python -u -m experiments.yahoo_downstream.questionnaire_reward \
  --em-root visual_results/yahoo_em/formal_bc_50rounds_v1 \
  --output visual_results/yahoo_questionnaire_reward/formal_v1 \
  --seeds 2022 2023 2024 --epochs 10 --steps-per-epoch 10000 \
  --alpha 0.1 --rating-threshold 3 --shuffle-seed 20261124 \
  --device cuda > yahoo_questionnaire_reward_formal.log 2>&1
```

输出目录必须不存在；不支持中断恢复。每epoch打印一次验证指标，完成显示 `Completed`。
运行时间以服务器首个epoch为准，较之前六次训练增加到九次。

## 输出及判断

- `config.json`：完整参数、输入和源码哈希。
- 每seed的 `reward_audit.json`：概率重建核验、门槛覆盖率、平坦问卷占比、辅助项幅度。
- `questionnaire_mapping.json`：用户与问卷 donor 对应，供复现打乱控制。
- `original/own/shuffled` 下的 `history.json`、`selection.json`、`best_dqn.pt`、
  `fixed_test.json` 和 `natural_test.json`，含逐用户结果和问卷/历史评分数量分组。
- `history.json` 另记录每epoch实际采到的辅助项均值、绝对值均值、非零步比例，
  区分“候选上存在辅助项”和“训练轨迹确实使用了辅助项”。
- `results.json`、`summary.json`、`paired_bootstrap.json`：本人−原始、打乱−原始、本人−打乱。

优先检查 own 是否同时优于 original 和 shuffled，并报告种子差异和集中度/覆盖率代价。
只有 own 优于 original 不能证明个体问卷匹配有价值；不能使用训练中含加分的reward做主结果。
用户bootstrap不覆盖模型训练或问卷打乱不确定性。固定10步为主要比较，自然退出为辅助解释。

完成后：

```bash
tar --exclude='*.pt' -czf yahoo_questionnaire_reward_results.tar.gz \
  visual_results/yahoo_questionnaire_reward/formal_v1 yahoo_questionnaire_reward_formal.log
```

## 本地验证

CPU，使用本机已有两轮EM子集产物。默认3分门槛下训练用户没有合格候选，三组退回相同reward，
这只验证零辅助项路径。另以2分门槛运行仅用于调试的短程检查，覆盖非零辅助项；正式预设仍是3分。
测试覆盖五题一致为零、手算期望和门槛、分池可复现打乱、alpha=0精确复现原始训练。
新增4项及旧下游5项测试通过；两个种子的短程流程完成，原始reward组逐用户结果精确复现
之前本地下游B组。非零辅助项路径确认有实际奖励贡献。实际服务器 CUDA/50轮产物需先执行 smoke。
