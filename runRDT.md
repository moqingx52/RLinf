# RoboTwin × RLinf：拿茶杯（place_empty_cup）RDT 后训练

本文档是 `run.md` 的 RDT 独立版本，覆盖 **RDT remote expert + DSRL steering** 训练链路。  
路径约定为 Docker/Linux（`/workspace/...`）。

/workspace# tree -L 2
.
|-- RLinf
|   |-- AGENTS.md
|   |-- CODE_OF_CONDUCT.md
|   |-- CONTRIBUTING.md
|   |-- LICENSE
|   |-- README.md
|   |-- README.zh-CN.md
|   |-- docker
|   |-- docs
|   |-- examples
|   |-- logs
|   |-- pyproject.toml
|   |-- ray_utils
|   |-- requirements
|   |-- rlinf
|   |-- rlinf.egg-info
|   |-- tests
|   |-- toolkits
|   `-- uv.lock
`-- RoboTwin
    |-- =0.6.11
    |-- =2.47.0
    |-- LICENSE
    |-- README.md
    |-- assets
    |-- code_gen
    |-- collect_data.sh
    |-- data
    |-- dataset
    |-- description
    |-- envs
    |-- eval_result
    |-- policy
    |-- robotwin
    |-- script
    |-- task_config
    |-- tasks
    |-- tools
    `-- weights



---

## 架构与管线（RDT）

推荐链路：`robotwin_rdt_server` + `robotwin_env_server` 在 RoboTwin 侧，RLinf 侧仅做 rollout/DSRL 训练并通过 TCP 调 expert。

| 组件 | 文件/配置 | 作用 |
|------|-----------|------|
| RoboTwin RDT 服务 | `RoboTwin/script/robotwin_rdt_server.py` | 对外提供与 DP server 兼容的 remote expert 协议（`init/dp_predict/dp_reset_history`）。 |
| RLinf 训练主配置 | `RLinf/examples/embodiment/config/robotwin_place_empty_cup_dsrl_rdt.yaml` | 指定环境、算法、batch、`expert_backend=remote`、RDT server 地址。 |
| RLinf 模型配置 | `RLinf/examples/embodiment/config/model/robotwin_rdt_dsrl.yaml` | 指定 `model_type=robotwin_rdt_dsrl`、checkpoint、`num_action_chunks` 等。 |
| RLinf 模型入口 | `RLinf/rlinf/models/embodiment/robotwin_rdt_dsrl/__init__.py` | 当前复用 DP policy 的 RL 壳子进行 DSRL + SAC 训练。 |

关键点：
- RLinf 端默认 remote，不在训练进程内直接加载 RoboTwin RDT 模型。
- `robotwin_rdt_server.py` 会加载 RDT 主模型、视觉编码器、文本编码器，并缓存 instruction embedding。
- server 的 `--n-action-steps` 控制每次只执行预测 chunk 的前 N 步，便于 chunk 级重规划。

---

## 1. 路径与关键参数

| 用途 | 典型路径（容器内） |
|------|-------------------|
| RLinf | `/workspace/RLinf` |
| RoboTwin | `/workspace/RoboTwin` |
| RDT 训练配置 | `/workspace/RLinf/examples/embodiment/config/robotwin_place_empty_cup_dsrl_rdt.yaml` |
| RDT 权重（示例） | `/workspace/RoboTwin/policy/RDT/checkpoints/place_empty_cup_only_170m/checkpoint-20000` |
| RDT base 配置 | `/workspace/RoboTwin/policy/RDT/configs/base_170m.yaml` |
| Vision encoder（默认） | `/workspace/RoboTwin/policy/weights/RDT/siglip-so400m-patch14-384` |
| Text encoder（默认） | `/workspace/RoboTwin/policy/weights/RDT/t5-v1_1-xxl` |

---

## 2. 环境变量与三卡 smoke test 分工

本轮目标是拉起一个最小但完整的 RDT 后训练 smoke test，固定使用物理 GPU `0/1/2`：

| 进程 | 端口/入口 | 物理 GPU | 环境 |
|------|-----------|----------|------|
| RDT expert server | `robotwin_rdt_server.py:8769` | `0` | `conda activate RoboTwin` |
| RoboTwin env server | `robotwin_env_server.py:8765` | `1` | `conda activate RoboTwin` |
| RLinf 后训练 smoke | `train_embodied_agent.py` | `2` | `source /workspace/RLinf/.venv/bin/activate` |

注意：`CUDA_VISIBLE_DEVICES=0/1/2` 应只写在各自启动命令或各自终端中，不要放到公共环境变量段里。设置后进程内看到的 `cuda:0` 会映射到该进程可见卡集合中的第一张物理卡。

```bash
cd /workspace/RLinf/examples/embodiment

export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export REPO_PATH=/workspace/RLinf
export ROBOTWIN_ROOT=/workspace/RoboTwin
export PYTHONPATH=/workspace/RLinf:${PYTHONPATH}

export ROBOTWIN_SERVER_ADDR=127.0.0.1:8765
export ROBOTWIN_RDT_SERVER_ADDR=127.0.0.1:8769
export ROBOTWIN_RDT_CKPT=/workspace/RoboTwin/policy/RDT/checkpoints/place_empty_cup_only_170m/checkpoint-20000
export RLINF_DP_EXPERT_BACKEND=remote

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

`robotwin_place_empty_cup_dsrl_rdt.yaml` 中 `actor.model.dp_server_addr` 默认读取 `ROBOTWIN_RDT_SERVER_ADDR`。

RDT 170M checkpoint 推荐传 checkpoint 目录即可，`robotwin_rdt_server.py` 会自动解析到 `pytorch_model/mp_rank_00_model_states.pt`，并在 checkpoint 路径包含 `170m` 时默认使用 `policy/RDT/configs/base_170m.yaml`。如果手动传 `--config`，以手动值为准。

推荐使用统一 gate 脚本执行下面所有阶段，避免绕过 smoke 验证直接长跑 SAC：

```bash
cd /workspace/RLinf/examples/embodiment

# 终端 1：RDT expert
./run_robotwin_rdt_post_training_gate.sh start-rdt-server

# 终端 2：RoboTwin env
./run_robotwin_rdt_post_training_gate.sh start-env-server

# 终端 3：第一道门控，remote RDT policy smoke eval
./run_robotwin_rdt_post_training_gate.sh smoke-eval

# 终端 3：第二道门控，RLinf 无噪声 rollout smoke
./run_robotwin_rdt_post_training_gate.sh zero-noise-smoke

# 终端 3：第三道门控，打开 DSRL/SAC 小规模学习
./run_robotwin_rdt_post_training_gate.sh dsrl-smoke
```

---

## 3. 启动 RoboTwin 服务（先于 RLinf）

启动顺序：`robotwin_rdt_server` -> `robotwin_env_server`。
```bash
docker exec -it rlinf-workspace /bin/bash
conda activate RoboTwin
```
### 3.1 渲染稳定性前置（建议）

```bash
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
unset VK_ICD_FILENAMES
unset VK_ADD_DRIVER_FILES
export VK_LOADER_DRIVERS_SELECT='*nvidia*'
export VK_LOADER_DRIVERS_DISABLE='*lvp*','*intel*','*radeon*','*virtio*'
```

### 3.2 RDT 推理服务

```bash
cd /workspace/RoboTwin

CUDA_VISIBLE_DEVICES=0 python script/robotwin_rdt_server.py \
  --host 0.0.0.0 \
  --port 8769 \
  --ckpt /workspace/RoboTwin/policy/RDT/checkpoints/place_empty_cup_only_170m/checkpoint-20000 \
  --config /workspace/RoboTwin/policy/RDT/configs/base_170m.yaml \
  --vision-encoder-path /workspace/RoboTwin/policy/weights/RDT/siglip-so400m-patch14-384 \
  --text-encoder-path /workspace/RoboTwin/policy/weights/RDT/t5-v1_1-xxl \
  --device cuda:0 \
  --instruction "Place the empty cup to the target area." \
  --n-action-steps 64
```

可选参数（按需）：
- `--config`：默认按 checkpoint 自动推断；170M checkpoint 会选 `policy/RDT/configs/base_170m.yaml`，否则选 `policy/RDT/configs/base.yaml`。
- `--vision-encoder-path`、`--text-encoder-path`：若权重不在默认目录必须显式传入。
- `--n-action-steps 64`：当前 RDT 后训练默认值；复现 RoboTwin 官方 RDT eval，等价于 `policy/RDT/deploy_policy.yml` 默认 `rdt_step=64`。RLinf 配置里的 `actor.model.num_action_chunks` 必须同步为 `64`。
- `--n-action-steps 4/8`：DSRL 稳定性 sweep 可尝试，但这已经不是官方 RDT eval 条件；RLinf 训练命令需同步覆盖 `actor.model.num_action_chunks`，并保证 `env.*.max_steps_per_rollout_epoch` 是该 chunk 长度的倍数。
- `--lora-adapter-path /path/to/adapter`：加载离线 LoRA 修补后的 RDT adapter 作为 remote expert。
- `--merge-lora`：加载后合并 LoRA 权重用于推理。
- `--control-frequency`：默认 `25`。
- `--no-log-ops`：关闭每次协议调用日志。

### 3.3 仿真环境服务

```bash
cd /workspace/RoboTwin

CUDA_VISIBLE_DEVICES=1 python script/robotwin_env_server.py \
  --host 0.0.0.0 \
  --port 8765 \
  --config /workspace/RLinf/examples/embodiment/config/env/robotwin_place_empty_cup.yaml \
  --assets-path /workspace/RoboTwin/ \
  --debug-level 2 \
  --debug-every 1
```

### 3.4 GPU 占用检查

三个进程都启动后，在宿主机或容器内执行：

```bash
nvidia-smi pmon -c 1
```

预期分布：
- `robotwin_rdt_server.py` 在物理 GPU `0`。
- `robotwin_env_server.py` 在物理 GPU `1`。
- `train_embodied_agent.py` 及其 Ray worker 在物理 GPU `2`。

如果 `robotwin_env_server.py` 仍出现在 GPU `0`，说明 SAPIEN/Vulkan 渲染没有被当前容器内的 `CUDA_VISIBLE_DEVICES=1` 完全限制；严格隔离时应把 env server 放到只暴露 GPU `1` 的容器/会话中启动，例如 Docker 层使用 `--gpus '"device=1"'`。

### 3.5 RDT 端到端 smoke eval

正式训练前建议先绕过 RLinf 训练器，直接用当前 RDT checkpoint 连 `robotwin_env_server` 跑 100 次完整 `place_empty_cup` episode。这个检查覆盖真实 observation、instruction、RDT action chunk 和 RoboTwin reward/success 判定。

```bash
cd /workspace/RLinf

ROBOTWIN_ROOT=/workspace/RoboTwin \
ROBOTWIN_SERVER_ADDR=127.0.0.1:8765 \
ROBOTWIN_RDT_SERVER_ADDR=127.0.0.1:8769 \
ROBOTWIN_RDT_CKPT=/workspace/RoboTwin/policy/RDT/checkpoints/place_empty_cup_only_170m/checkpoint-20000 \
python toolkits/eval_scripts_robotwin/eval_remote_rdt_smoke.py \
  --episodes 100 \
  --seed 0 \
  --config /workspace/RLinf/examples/embodiment/config/env/robotwin_place_empty_cup.yaml \
  --task-name place_empty_cup \
  --step-lim 500
```

当前后训练默认也使用 `--n-action-steps 64`，所以 smoke eval 和训练应保持同一个 RDT server 配置。RDT 语义是固定用 `img_history_size=2` 的观测窗口预测 `action_chunk_size=64`，再取前 `rdt_step/--n-action-steps` 个动作执行；不是“观测到多少 step 就取多少动作”。这里重点验证 remote server 的 history warmup、reset、相机顺序和 action chunk 执行是否与官方 deploy 对齐。

注意：官方 `policy/RDT/deploy_policy.py` 在执行 64 个动作时，每个 inner action 后都会更新 observation window；remote TCP env 当前只返回整块动作执行后的最终 obs，无法拿到倒数第二帧。`robotwin_rdt_server.py` 已采用非 stale 近似：reset 后用 `[None, current]`，后续用 `[current, current]`，避免把上个 chunk 起点误当上一帧。如果 100 次仍为 `0/100`，下一步应让 `robotwin_env_server` 返回 chunk 内最后两帧，再把 RDT server history 完全对齐官方 deploy。

输出会逐 episode 打印 `success`、`return`、`steps`、`take_action_cnt` 和最后一步 `reward_components/reward_milestones`，并在 `RLinf/logs/robotwin-rdt-smoke-eval/` 写入 jsonl 与 summary。若这里接近你离线 eval 的成功率，而训练指标仍长期为 0，应优先排查训练侧 reset/history、noise 注入或 reward 聚合。

---

## 4. RLinf 后训练 smoke（DSRL + 冻结 RDT）

这一节启动一个完整的 RLinf 后训练 smoke：RLinf 会连上两个 RoboTwin 服务，采集 rollout，写入 replay buffer，执行 SAC/Q 更新，并按配置保存 checkpoint。它不是最终收敛训练，而是验证“RDT 能产生可奖励轨迹、DSRL 头能参与学习、训练闭环能跑完”。

后训练机制简述：
- RDT 主模型冻结，仍由 `robotwin_rdt_server` 推理，不在 RLinf 训练进程里加载。
- RLinf 训练的是一个轻量 DSRL steering 头，它根据当前图像和机器人状态输出 RDT diffusion 的 `init_noise`。
- `robotwin_rdt_server` 用这个 `init_noise` 生成 64 步动作 chunk，`robotwin_env_server` 执行动作并返回 reward/success/debug trace。
- SAC critic 学习“什么样的 `init_noise` 更容易触发 grasp/lift/place/success”，actor 再朝高 Q 的噪声方向更新。
- 配置默认 `dsrl_noise_scale=0.0`，先强制 RLinf rollout 复现冻结 RDT expert；只有采到非零 reward / success 后，才在第三阶段显式切到 `0.05` 做 DSRL/SAC 学习。

启动命令：

```bash
docker exec -it rlinf-workspace /bin/bash
#退出base环境
conda deactivate
#启动uv环境
source .venv/bin/activate
```
### 4.1 三卡无噪声 smoke（物理 GPU 0/1/2）

下面命令默认你已经执行完第 2 节环境变量导出，并已启动第 3 节两个服务：

```bash
cd /workspace/RLinf/examples/embodiment

ROBOTWIN_RDT_CKPT=/workspace/RoboTwin/policy/RDT/checkpoints/place_empty_cup_only_170m/checkpoint-20000 \
ROBOTWIN_RDT_SERVER_ADDR=127.0.0.1:8769 \
CUDA_VISIBLE_DEVICES=2 python train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config/ \
  --config-name robotwin_place_empty_cup_dsrl_rdt \
  runner.logger.log_path=/workspace/RLinf/logs/dsrl-rdt-place-empty-cup-zero-noise-smoke \
  runner.max_epochs=10 \
  runner.save_interval=1 \
  +runner.keep_last_checkpoints=3 \
  env.train.video_cfg.save_video=false \
  env.eval.video_cfg.save_video=false \
  env.train.total_num_envs=1 \
  env.eval.total_num_envs=1 \
  actor.micro_batch_size=8 \
  actor.global_batch_size=8 \
  rollout.enable_offload=false \
  actor.enable_offload=false \
  actor.fsdp_config.gradient_checkpointing=true \
  rollout.collect_prev_infos=false \
  actor.model.dsrl_noise_scale=0.0 \
  actor.model.num_action_chunks=64 \
  env.train.max_steps_per_rollout_epoch=512 \
  env.eval.max_steps_per_rollout_epoch=512
```

测试重点：
- 确认训练进程、Ray worker 都只出现在物理 GPU `2`。
- 确认能连通 `robotwin_rdt_server` 和 `robotwin_env_server`，能进入 rollout，并能打印 reward/action trace。
- 每个 epoch 是一个 512-step rollout，刚好覆盖 `place_empty_cup` 的 500-step episode 上限；`runner.max_epochs=10` 约等价于跑 10 个完整训练 episode，并在每个 epoch 后执行轻量 SAC 更新。
- 这个阶段必须保持 `actor.model.dsrl_noise_scale=0.0`。如果这里仍长期 `reward=0/success_once=0`，不要进入 DSRL 正式学习，先排查 RLinf rollout 的 history reset、reward 聚合、action chunk、camera/wrist 配置。
- `+runner.keep_last_checkpoints=3` 使用 Hydra append 语法；当前配置是 struct 模式，写成 `runner.keep_last_checkpoints=3` 会报 `Key 'keep_last_checkpoints' is not in struct`。

### 4.2 DSRL smoke（通过无噪声门控后）

无噪声 RLinf smoke 采到非零 reward / success 后，再打开小扰动和轻量 SAC 更新：

```bash
cd /workspace/RLinf/examples/embodiment

./run_robotwin_rdt_post_training_gate.sh dsrl-smoke
```

等价关键覆盖：

```bash
actor.model.dsrl_noise_scale=0.05 \
algorithm.update_epoch=5 \
algorithm.train_actor_steps=1 \
algorithm.entropy_tuning.target_entropy=auto_half_action_dim
```

### 4.3 后续正式训练（可选，物理 GPU 2-7）

remote RDT smoke、无噪声 RLinf smoke 和 DSRL smoke 都通过后，再考虑启动 6 卡正式训练：

```bash
cd /workspace/RLinf/examples/embodiment

ROBOTWIN_RDT_CKPT=/workspace/RoboTwin/policy/RDT/checkpoints/place_empty_cup_only_170m/checkpoint-20000 \
ROBOTWIN_RDT_SERVER_ADDR=127.0.0.1:8769 \
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config/ \
  --config-name robotwin_place_empty_cup_dsrl_rdt \
  runner.logger.log_path=/workspace/RLinf/logs/dsrl-rdt-place-empty-cup-6gpu \
  +runner.keep_last_checkpoints=3 \
  env.train.video_cfg.save_video=false \
  env.eval.video_cfg.save_video=false \
  env.train.total_num_envs=6 \
  env.eval.total_num_envs=6 \
  actor.micro_batch_size=8 \
  actor.global_batch_size=48 \
  rollout.enable_offload=false \
  actor.enable_offload=false \
  actor.fsdp_config.gradient_checkpointing=true \
  rollout.collect_prev_infos=false \
  actor.model.dsrl_noise_scale=0.05 \
  algorithm.entropy_tuning.target_entropy=auto_half_action_dim \
  algorithm.update_epoch=20 \
  algorithm.train_actor_steps=2 \
  env.train.max_steps_per_rollout_epoch=512 \
  env.eval.max_steps_per_rollout_epoch=512
```

说明：
- `+runner.keep_last_checkpoints=3` 只保留最近 3 个 `global_step_*` checkpoint，旧 checkpoint 会在新 checkpoint 保存成功后自动删除。这里必须使用 `+`，否则 Hydra struct 配置会拒绝新增字段并报 `ConfigAttributeError`。
- `env.train.video_cfg.save_video=false` 和 `env.eval.video_cfg.save_video=false` 关闭训练/评估视频落盘，避免 log 目录继续膨胀。
- `actor.global_batch_size=48` 对齐 6 张训练卡和 `actor.micro_batch_size=8`，避免分布式 batch 不整除。
- RDT smoke/正式训练默认保持 offload 关闭。当前 `actor.fsdp_config.use_orig_params=true` 时，开启 `actor/rollout.enable_offload=true` 在部分版本可能触发 FSDP writeback 的 shape mismatch（`Expects [flat] but got [matrix]`）。

### 4.4 如何判断 smoke 有效

优先看 RoboTwin env server 的 debug 日志和 RLinf TensorBoard：
- env server 日志里出现 `reward_components.events` 的 `grasp`、`lift`、`place`、`release`、`success`，说明模型在后训练 rollout 中做出了有奖励的正确片段。
- RLinf 日志里 `episode/success_once` 或 `success_at_end` 出现 `true`，说明训练链路采到了完整成功例子。
- 即使暂时没有 `success`，只要 `grasp/lift/place` 里程碑开始出现，critic 就已经有非零奖励信号可学。
- `q_data`、actor loss、alpha/entropy 不报 NaN，checkpoint 正常保存，说明 SAC 更新闭环已跑通。

### 4.5 DSRL sweep 建议

**64-step baseline**：保持 RDT server `--n-action-steps 64`，训练配置默认 `actor.model.num_action_chunks=64`。这是当前 smoke eval 已经出现成功样本的条件，优先用于后训练修补。

默认 `actor.model.dsrl_noise_scale=0.0` 用于门控验证；通过 remote RDT smoke 和无噪声 RLinf smoke 后，再显式覆盖为 `0.05` 或做小范围 sweep。

**8-step 稳定性对照**：RDT server 改为 `--n-action-steps 8`，RLinf 命令同步加：

```bash
actor.model.num_action_chunks=8 \
env.train.max_steps_per_rollout_epoch=512 \
env.eval.max_steps_per_rollout_epoch=512 \
runner.logger.experiment_name=robotwin_dsrl_rdt_8step
```

**4-step 稳定性对照**：RDT server 改为 `--n-action-steps 4`，RLinf 命令同步加：

```bash
actor.model.num_action_chunks=4 \
env.train.max_steps_per_rollout_epoch=512 \
env.eval.max_steps_per_rollout_epoch=512 \
runner.logger.experiment_name=robotwin_dsrl_rdt_4step
```

`algorithm.entropy_tuning.target_entropy` 支持：
- `auto_action_dim`：按完整 RDT latent noise 维度取 `-action_dim`。
- `auto_half_action_dim`：按完整 RDT latent noise 维度取 `-0.5 * action_dim`，当前默认。
- 数值：例如 `-224`、`-448`、`-896`，用于正式 sweep。

### 4.6 Reward trace

`place_empty_cup` 当前使用单调里程碑 reward：
- `grasp`：夹爪接触杯子且对应夹爪不是 open，首次 `+0.5`。
- `lift`：杯子相对初始高度达到 `0.045m`，首次 `+1.0`。
- `place`：杯子功能点接近杯垫功能点，首次 `+2.0`。
- `release/success`：沿用 `check_success()`，首次 `+5.0` 并终止。
- approach/place 仅保留正向 shaping，退步不给负分。

训练配置默认打开 `debug_reward_trace` 和 `debug_action_trace`，RoboTwin server `--debug-level 2` 会在 `debug_trace.info_focus.reward_components.events` 中打印最近一步触发的里程碑事件。

### 阶段3（预留）

离线 LoRA 修补：使用 DSRL 成功轨迹、失败纠正轨迹和原始 SFT demo 混合做小步 MSE/SFT，保存 PEFT adapter。推理时启动：

```bash
CUDA_VISIBLE_DEVICES=0 python script/robotwin_rdt_server.py \
  --host 0.0.0.0 \
  --port 8769 \
  --ckpt /workspace/RoboTwin/policy/RDT/checkpoints/place_empty_cup_only_170m/checkpoint-20000 \
  --lora-adapter-path /workspace/RoboTwin/policy/RDT/checkpoints/place_empty_cup_lora_adapter \
  --merge-lora \
  --device cuda:0 \
  --instruction "Place the empty cup to the target area." \
  --n-action-steps 64
```

### 阶段4（预留）

（待填）

---

## 5. 与 DP 路线的关键差异

- 训练配置：`robotwin_place_empty_cup_dsrl_rdt`（不是 `..._dsrl_dp`）。
- expert 服务：`robotwin_rdt_server`（默认端口 `8769`），而不是 `robotwin_dp_server`（常见 `8767`）。
- RDT 后训练默认设置：`num_action_chunks=64`、`n_obs_steps=2`、`action_dim=14`。
- server 内部预测 horizon 由 RDT config 的 `action_chunk_size` 决定，`--n-action-steps` 只截取前 N 步用于执行。

---

## 6. 常见问题（RDT）

| 现象 | 处理 |
|------|------|
| 训练端连不上 expert | 检查 `robotwin_rdt_server` 是否启动，核对 `ROBOTWIN_RDT_SERVER_ADDR`。 |
| `init` 阶段失败 | 检查 `ROBOTWIN_RDT_CKPT` 是否存在且可读。 |
| 编码器加载失败 | 显式传 `--vision-encoder-path` / `--text-encoder-path` 到本机真实路径。 |
| `ConfigAttributeError: Key 'keep_last_checkpoints' is not in struct` | 使用 `+runner.keep_last_checkpoints=3`，不要写成 `runner.keep_last_checkpoints=3`。 |
| `RuntimeError: Cannot writeback when the parameter shape changes` | 先确认 smoke/正式训练命令保持 `actor.enable_offload=false` 与 `rollout.enable_offload=false`；若需开启 offload，请同步使用包含 FSDP 参数迁移修复的 RLinf 版本。 |
| batch size 被锁后报错 | `dp_predict` 的 batch 维要与同连接首次推理一致；变更并行度需重启客户端连接。 |
| selective reset 报错 | 需先至少一次 `dp_predict` 建立 batch，再做带 `env_mask/env_idx` 的 `dp_reset_history`。 |

---

## 7. 排查建议

- 先按三卡 smoke 分工启动：GPU `0` 跑 RDT server，GPU `1` 跑 RoboTwin env server，GPU `2` 跑 RLinf 后训练 smoke。
- 三卡 smoke 通过后，再考虑使用物理 GPU `2-7` 跑 6 env 正式训练。
- 正式训练保持 `actor.micro_batch_size=8`、`actor.global_batch_size=48`，即 `8 * 6`。
- 重点看 env server 日志里的 `reward_components.events`：如果 1-2k macro steps 后没有 `grasp`，优先排查动作语义/RDT server 输出；如果有 `grasp/lift` 但没有 `place/success`，再调 reward 阈值或 chunk 步长。
