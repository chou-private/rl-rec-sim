# 独立记录网络小检验：global / own / shuffled

入口 `python -m experiments.yahoo_recording.run`，只需 NumPy/PyTorch 和 Yahoo 原始数据。
不读取已有 EM checkpoint，不训练 DeepFM，不运行 DQN。这是对记录统计的独立预测检验。

## 为什么需要共同的星级分布模型

只观察主动五星数量，无法区分“本来就更喜欢歌曲”和“五星更愿意记录”。因此先拟合一个
共同的辅助模型 p_v(Q)=P(潜在星级=v | Q)，只用随机收集评分，随后冻结。
所有三组都使用**本人问卷对应的同一个 p(Q)**，包括 global 与 shuffled。
shuffled 只打乱记录网络的问卷输入，不打乱潜在喜好分布；否则无法隔离记录机制的增量。
这里的 global 指记录概率全局共享，并不表示整个统计模型完全不看问卷。

## 训练目标

每人1000首歌，历史主动评分汇总为五个计数 a_uv，其余为历史表中未记录数量。
随机评分另外汇总为五个计数 b_uv（每人共10条），不把随机评分当作未记录负例。

1. 星级辅助模型：五维问卷 `(Q2..Q6-3)/2` → 线性 softmax（5类）。以全局随机边际初始化，
   仅校准用户的随机评分计数参与梯度更新。损失是随机评分交叉熵 + .01×线性权重平方均值。
   验证用户随机评分选择 checkpoint，包括零权重的初始全局边际。随后冻结。
2. global：五个常数 mu_v。先由校准用户主动/随机计数比值初始化，再在共同 p(Q) 下，
   以校准主动记录的六分类计数似然优化五个参数，验证主动统计选择 checkpoint。
   因此这是**重新拟合的全局概率对照**，不是直接复用之前 EM 的固定五个数。
3. own / shuffled：`mu_uv = sigmoid(logit(mu_global_v) + h(Q)_v)`。
   h为5→8→5、Tanh小网络，输出层零初始化，全局基准固定。损失为计数NLL + .001×修正logit平方均值。
   两组同seed初始化相同。保留epoch0，即允许验证集选择完全退回global。

六种观测结果的概率：

```
P(历史记录为v星 | Q) = p_v(Q) * mu_v(Q)
P(历史未记录 | Q) = 1 - sum_v p_v(Q) * mu_v(Q)
```

目标为这六类的计数负对数似然，除以1000对用户等权；省略与模型无关的多项式系数。
全批次Adam，lr=.01，最多400轮，验证NLL连续30轮无改善停止。
问卷网络只读取Q2—Q6，**不读取用户ID、测试主动数量、测试随机星级或历史数量作为输入**。

注意：这里把用户内歌曲视为可交换，并用共享问卷条件概率近似记录过程；无法完全处理
同一回答内部的喜好/记录相关性、历史与随机评分时点差异、发布删失、至少10条历史记录的人群筛选。
计数模型是工作假设。降低NLL不等于估计出了真实个人记录概率，星级辅助模型误差也可能被记录网络吸收。
只能称“对留出记录统计的解释/预测能力”，不能称“记录概率准确率”或因果效果。

## 划分、控制和评价

- 复用split seed=20261123，5400用户按问卷模式分层为3238校准、1078验证、1084测试。
- 只用校准标签做梯度更新；验证用于选择两个阶段的checkpoint，测试只在选择后评估。
- 之前已多次探索同一数据，本实验是内部留出诊断，不声称全新独立外部验证。
- 固定shuffle seed=20261124，在各原用户池内打乱完整五题向量。映射跨模型seed保持一致。
  打乱控制只涉及mu网络，p(Q)仍与本人对应。单一排列不构成完整随机化检验。
- 全局模型和辅助星级模型只拟合一次，共用于三个网络seed；不可把重复输出的global视为三个独立训练。

主要指标：留出用户的 `record_nll`（含缺失类，越低越好）。同时检查：

- `total_count_mae`：预计总记录数的误差；
- `star_count_mae`：各星级主动记录数的平均绝对误差；
- `recorded_star_nll`：条件于已记录后的星级交叉熵，不包含记录总量。

若只改善总记录数而不改善星级构成，不能说发现了更好的星级选择偏差修正。
若本人不能同时优于global和shuffled，则不据此继续接EM/DQN。
辅助模型另报告随机评分上的global和conditional NLL，检查喜好估计是否合理。
用户配对bootstrap区间不包含辅助模型、记录网络和打乱排列的不确定性。

## 服务器命令

在仓库根目录、原来的NumPy/PyTorch环境中，切换 `codex/yahoo-recording-network`。
这个模型很小，推荐CPU；不需要50轮EM的 `.pt` 或 `.npy` 文件。

```bash
python -m unittest discover -s tests -p test_yahoo_recording.py -v

python -u -m experiments.yahoo_recording.run \
  --data data/YahooR3/data_raw \
  --output visual_results/yahoo_recording_network/server_smoke \
  --smoke-users-per-group 20 --epochs 50 --patience 10 \
  --seeds 2022 2023 --device cpu \
  > yahoo_recording_network_smoke.log 2>&1
```

看到 `Completed` 后正式检验：

```bash
python -u -m experiments.yahoo_recording.run \
  --data data/YahooR3/data_raw \
  --output visual_results/yahoo_recording_network/formal_v1 \
  --seeds 2022 2023 2024 --epochs 400 --patience 30 \
  --hidden 8 --lr 0.01 --shrinkage 0.001 --marginal-l2 0.01 \
  --split-seed 20261123 --shuffle-seed 20261124 --device cpu \
  > yahoo_recording_network_formal.log 2>&1
```

输出目录必须不存在，不覆盖旧结果。每完成一个小模型打印一条日志。
400轮是小网络的全批次优化轮数，不是400轮EM，也不是DQN交互轮数。

## 输出

- `config.json`、`data_audit.json`、`splits.json`、`questionnaire_mapping.json`：参数与数据审计。
- `marginal_selection.json`、`marginal_test.json`：共同星级分布模型的验证选择、留出随机评分表现。
- `global_selection.json`、`global_probability.json`：全局记录模型训练及初始化/最终概率。
- 每seed的 `own_selection.json`、`shuffled_selection.json`：轮次及完整训练曲线。
- 每seed的 `global_test.json`、`own_test.json`、`shuffled_test.json`：逐用户/组指标、概率边界情况。
- `results.json`、`summary.json`、`paired_bootstrap.json`：三组比较和用户配对区间。

打包：

```bash
tar --exclude='*.pt' -czf yahoo_recording_network_results.tar.gz \
  visual_results/yahoo_recording_network/formal_v1 yahoo_recording_network_formal.log
```

## 本地检查

6项单元检查：手算六类似然和计数、概率归一化、零修正退回全局、随机星级监督、
合成数据恢复已知记录概率、epoch0回退、学习合成问卷条件信号（部分项目合并为同一测试）。
100人、两个网络seed、最多50轮的小规模CPU流程完成；不把其结果当成正式研究证据。
