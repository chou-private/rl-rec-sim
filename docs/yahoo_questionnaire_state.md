# Q2—Q6 作为 DQN 状态：zero / own / shuffled

入口：`python -m experiments.yahoo_downstream.questionnaire_state`。
只依赖 NumPy / PyTorch，复用完整50轮 EM 的 B 模型、原始 reward 和现有下游评估。
不重新训练 EM，不叠加任何问卷辅助reward，不使用 C 的预测反馈。

## 固定实验设计

- 三组同一输入维度：原状态（冻结 B 用户向量 + 最近三步歌曲/反馈平均）后拼接五维。
- `zero`：五维全0；`own`：本人 Q2—Q6，`(answer-3)/2` 映射为[-1,1]；
  `shuffled`：完整五维答案向量在各数据池内固定打乱，不打乱题目。
- 不做用户内中心化。全5回答保留为全1，全1保留为全-1，全3才是全0。
  这是有序问卷的数值编码，不是概率，也不保证一定提供增量信息。
- 相同 seed 下三组网络结构、参数数量、初始化均一致；零组也有五个额外输入。
  因此必须重跑 zero，不能直接把旧无五维输入的 B+DQN 当严格对照。
- 默认固定 shuffle seed=20261124，原 calibration/validation/test 池内分别打乱完整向量，
  不按问卷组分层打乱。映射在训练、验证、测试全过程固定，各模型seed共用映射；没有逐步重抽。
- 问卷只改变状态。三组训练奖励均为 `(B期望评分-1)/4`，`alpha=0`、无辅助项。
- 三组共用冻结的 B 用户/歌曲嵌入；问卷本身作为已知静态协变量，在每步和初始状态都可用。
  测试用户问卷可以使用，测试评分/收益不能用于构造特征。假设部署前已经取得问卷。
- 训练用 calibration 用户，验证按固定10步共同 pseudo-GT reward 选择检查点，测试只在选完后运行。
- 其余预算与上一轮一致：10epoch×10000步、三个配对种子、Double DQN、gamma=.9、最近三步、
  会话内禁止重复、自然退出训练、固定10步/自然退出评估。参见 `docs/yahoo_em_downstream.md`。
- 三种子是 EM 与 DQN 整条流程的重复，不把相同用户跨种子视为独立样本。

## 服务器命令

切到 `codex/yahoo-questionnaire-state`，在仓库根目录激活原 PyTorch 环境。
输入必须是包含 `.pt`、`.npy` 的完整 EM 目录；轻量结果压缩包不能代替它。

```bash
python -m unittest discover -s tests -p test_yahoo_questionnaire_state.py -v

python -u -m experiments.yahoo_downstream.questionnaire_state \
  --em-root visual_results/yahoo_em/formal_bc_50rounds_v1 \
  --output visual_results/yahoo_questionnaire_state/server_smoke \
  --seeds 2022 --smoke-users 5 --epochs 2 --steps-per-epoch 100 \
  --learning-starts 20 --batch-size 16 --hidden 16 --capacity 500 \
  --device cuda > yahoo_questionnaire_state_smoke.log 2>&1
```

出现 `Completed. Results: ...` 后正式运行（已有会话，无需 nohup）：

```bash
python -u -m experiments.yahoo_downstream.questionnaire_state \
  --em-root visual_results/yahoo_em/formal_bc_50rounds_v1 \
  --output visual_results/yahoo_questionnaire_state/formal_v1 \
  --seeds 2022 2023 2024 --epochs 10 --steps-per-epoch 10000 \
  --shuffle-seed 20261124 --device cuda \
  > yahoo_questionnaire_state_formal.log 2>&1
```

输出目录必须是新目录，不覆盖旧实验；不支持断点恢复。默认九次 DQN 训练，每epoch打印一次。
查看进度：`tail -f yahoo_questionnaire_state_formal.log`。

## 文件及结果解释

- `config.json` 保存参数、源码及数据哈希；每seed保存 EM artifact 审计和实际用户池。
- `questionnaire_mapping.json` 保存原始用户与 donor 用户对应。
- 各组 `state_features.npy` 与 `feature_audit.json` 保存实际五维输入、哈希、范围、非零比例。
- `best_dqn.pt` 保存网络与状态维度；重载时必须配套同一 B 嵌入、用户映射及 state_features.npy，
  不能单独把网络当作旧 DQN 加载。本入口在评估前已经重新加载检查点进行验证。
- 各组 `history.json`、`selection.json`、`fixed_test.json`、`natural_test.json`。
  日志中的辅助reward统计三组必须为0。
- `results.json`、`summary.json`、`paired_bootstrap.json`：本人−全零、打乱−全零、本人−打乱。
  主要指标为固定10步平均收益，同时报告覆盖率/集中度和自然退出。用户区间不覆盖训练或排列不确定性。

本人组应同时优于全零和打乱，才支持个体问卷匹配的增量价值；不因单个种子改善就宣称有效。
固定随机排列只是一个对照，不是完整的随机化检验。模拟环境结果也不能代表线上满意度。
本轮不增加额外编码/题目/奖励搜索；若未发现稳定增量，按预定范围报告结果。

完成后打包：

```bash
tar --exclude='*.pt' --exclude='*.npy' \
  -czf yahoo_questionnaire_state_results.tar.gz \
  visual_results/yahoo_questionnaire_state/formal_v1 yahoo_questionnaire_state_formal.log
```

## 本地检查

CPU、已有两轮 EM 子集产物：三个状态组、两个种子，每组200交互步，验证保存重载和统一评估。
单元测试覆盖绝对回答信息保留、三组维度相同、历史表示不变、问卷在初始及后续状态固定、
分池打乱可复现、非零问卷输入列确实得到梯度更新、零输入列不更新，以及修改测试reward不影响训练选模。
旧下游/问卷reward回归检查也需通过。CUDA和完整50轮EM产物仍以服务器smoke为准。
