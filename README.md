# 松灵 Piper + LeRobot 自主抓取 · 本地环境全套脚本

这套脚本把你从零到“Piper 自主抓取”的完整流程串起来，针对 **RTX 5060（8GB 显存）** 做了显存优化。

## 文件说明

| 文件 | 作用 |
|---|---|
| `config.env` | 全局配置（CAN、相机、遥操作设备、数据集、训练参数），**先改这里** |
| `00_check_env.sh` | 环境自检（只读，跑一遍看缺什么） |
| `01_setup_can.sh` | 配置并激活 Piper 的 CAN 总线（需 sudo） |
| `02_install_env.sh` | 一键安装 miniforge + conda 环境 + PyTorch(cu128) + LeRobot + Piper 插件 |
| `03_test_piper.py` | 只读测试：验证 CAN + piper_sdk 通信是否正常 |
| `04_record.sh` | 遥操作采集抓取示范数据集 |
| `05_train_act.sh` | 训练 ACT 策略（5060 优化） |
| `06_deploy.sh` | 部署策略，让 Piper 自主抓取 |

## 前置条件

- **机械臂**：AgileX Piper + 原厂 USB-CAN 模块 + 末端夹爪
- **相机**：一个 USB 相机（对着工作台/腕部）
- **遥操作设备（三选一）**：SO-101 主臂 / SpaceMouse / PiperMate 主臂（见第 5 节）
- **系统**：Ubuntu 22.04 / 24.04，x86_64
- **显卡**：RTX 5060（Blackwell），NVIDIA 驱动 **≥ 570**（建议升到最新 580+）

## 快速开始（按顺序执行）

```bash
cd /home/mage/piper_ws/lerobot_setup
chmod +x *.sh

# 1) 先看环境缺什么
bash 00_check_env.sh

# 2) 激活 CAN（机械臂上电 + USB-CAN 插好）
bash 01_setup_can.sh

# 3) 安装整套环境（首次约 10~20 分钟）
bash 02_install_env.sh

# 4) 验证机械臂 CAN 通信
python 03_test_piper.py

# 5) 采集抓取示范数据（先 50 条）
bash 04_record.sh

# 6) 训练 ACT
bash 05_train_act.sh

# 7) 部署抓取（先改 config.env 的 POLICY_PATH）
bash 06_deploy.sh
```

> 每次开新终端都要先激活环境：`conda activate lerobot`

## 1. 改配置

打开 `config.env`，按实际改这几项：

- `CAMERA_JSON`：相机索引 `index_or_path`（多相机用 `ls /dev/video*` 确认）
- `TELEOP_TYPE` / `TELEOP_ARGS`：你的遥操作设备（见第 5 节）
- `TASK_DESCRIPTION`：写清楚要抓什么，例如 `"grab the red cube and place it into the box"`
- `NUM_EPISODES`：先 `50`，流程跑通后再加到 100+

## 2. CAN 总线说明

Piper 通过 **CAN 总线**（1 Mbps）通信，AgileX USB-CAN 模块插上后是 `can0`。
`01_setup_can.sh` 会自动：装 `can-utils/ethtool/iproute2` → 找到 CAN 接口 → 统一改名 `can0` → 设 1 Mbps → 激活。

> **每次重启电脑、重插 USB-CAN 后，都要重新运行 `bash 01_setup_can.sh`。**

## 3. 训练显存优化（5060 8GB）

`05_train_act.sh` 已按 8GB 优化（`batch_size=4`）。如果训练时 OOM（显存不足）：

1. 把 `BATCH_SIZE` 降到 `2`
2. 把 `CAMERA_JSON` 分辨率降到 `320x240`
3. 训练时关掉浏览器/游戏等占用显存的程序，用 `nvidia-smi` 或 `nvitop` 盯着

ACT 在 5060 上抓取任务通常 **30 分钟 ~ 2 小时** 出结果。想跑 Diffusion/VLA 才需要租 4090。

## 4. 训练步数怎么定

`STEPS` ≈ `epochs × (总帧数 / batch)`。

- 50 条 × 45s × 30fps ≈ 6.75 万帧，batch=4 时 1 epoch ≈ 1.7 万步，10 epoch ≈ 17 万步。
- 小数据集建议先 `STEPS=50000` 看抓取效果，效果不够再逐步加。

## 5. 遥操作设备选择

采集示范数据需要“主臂”来拖动/操控 Piper。三选一，改 `config.env`：

| 设备 | TELEOP_TYPE | 说明 |
|---|---|---|
| **SO-101 主臂**（推荐，最常见） | `so101_leader` | LeRobot 原生支持，`pip install 'lerobot[feetech]'` |
| **SpaceMouse** | `spacemouse` | 需 `pip install lerobot-teleoperator-spacemouse` |
| **PiperMate 主臂** | `pipermate` | 需 FashionStar 插件 `pip install lerobot-teleoperator-pipermate` |

如果没有主臂，最省事的是买/借一个 SO-101 主臂，或用 SpaceMouse 做笛卡尔遥操作。

## 6. 部署时找不到策略？

训练结束后检查点在 `$OUTPUT_DIR/$JOB_NAME/checkpoints/` 下（通常有 `last/pretrained_model` 或按 step 数字命名的目录）。用 `ls` 看一眼，把 `config.env` 的 `POLICY_PATH` 指过去即可。

## 常见问题

- **`no kernel image is available for sm_120`** → 驱动太老，升级到 570+ 后重装 cu128 PyTorch。
- **报某参数不存在** → 各版本 CLI 参数可能略有差异，运行 `lerobot-record --help` / `lerobot-train --help` 对照调整。
- **采集时机械臂不动** → 检查 `bash 01_setup_can.sh` 是否成功、`candump can0` 是否有数据、机械臂是否已上电且不是“主臂模式”。
- **数据采集后训练找不到数据集** → 确认 `04` 和 `05` 的 `DATASET_ROOT` / `DATASET_REPO_ID` 一致。

## 参考

- [LeRobot 官方仓库](https://github.com/huggingface/lerobot)
- [LeRobot 安装文档](https://huggingface.co/docs/lerobot/installation)
- [LeRobot Compute 硬件指南](https://huggingface.co/docs/lerobot/hardware_guide)
- [Piper 插件 lerobot_robot_piper (AgRobotics)](https://github.com/AgRoboticsResearch/lerobot_robot_piper)
- [AgileX piper_sdk](https://github.com/agilexrobotics/piper_sdk) / [piper_sdk_demo（含 CAN 配置）](https://github.com/agilexrobotics/piper_sdk_demo)
