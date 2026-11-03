# 松灵 Piper + LeRobot 自主抓取（仿真 + 真机）

用 **LeRobot ACT（动作分块模仿学习）** 控制松灵 **Piper** 六轴机械臂完成「抓取物体并抬起」。
项目包含 **MuJoCo 仿真** 与 **真机** 两套完整链路：遥操作采集 → 训练 ACT → 部署推理。

- 仿真：键盘 / Xbox 手柄笛卡尔遥操作，验证整条软件链路（**抓取成功率 5/5**）
- 真机：piper_sdk + D435i 相机，cv2 控制面板采集 **30 条示范**，训练 ACT 后自主抓取
- 附带数据后处理（动作 EMA 平滑）与训练曲线可视化

> 核心是**模仿学习**（监督回归），不是强化学习：没有 reward、没有 value function。

---

## 一、环境要求

| 项 | 要求 |
|---|---|
| 系统 | Ubuntu 24.04（x86_64） |
| 显卡 | NVIDIA RTX 5060（8GB），驱动 ≥ 570 |
| Python | 3.12（conda 环境 `lerobot`） |
| 关键依赖 | torch cu130、lerobot 0.6.1、mujoco、piper_sdk（V2 协议）、pyrealsense2 |
| 真机 | 松灵 PiPER（CAN 接口 can0）+ Intel RealSense D435i |

```bash
conda activate lerobot
cd /home/mage/piper_ws/lerobot_setup
```

`common.sh` 里的 `sanitize_env()` 会清掉 `LD_LIBRARY_PATH` 和代理变量（否则会触发
`cublasLtCreate` 崩溃和 httpx 代理错误）。所有脚本都通过它跑。

---

## 二、快速开始

### A. 仿真链路（无需硬件）

```bash
# 1. 采集（手柄或键盘遥操作，采 30 条）
python sim_collect_piper.py --num_episodes 30 --overwrite --controller gamepad

# 2. 训练 ACT
bash train_piper.sh 20000

# 3. 部署（MuJoCo viewer + 策略视角相机）
bash rollout_piper.sh

# 4. 画 loss 曲线
python plot_training.py ~/lerobot_outputs/act_piper_keyboard.log
```

**手柄按键**：左摇杆平移、右摇杆升降、A 夹爪开合、B 丢弃、Start 下一条、Select 退出。

### B. 真机链路

```bash
# 0. 一次性检查硬件（CAN + 相机）
bash 00_check_env.sh
bash 01_setup_can.sh          # 配置 can0 为 1Mbps

# 1. 采集（cv2 控制面板：START/END/HOME/DISCARD/QUIT）
python real_collect.py --repo_id local/piper_real --overwrite

# 2. 训练
bash train_piper.sh 20000 local/piper_real

# 3. 部署（START/END/EMERGENCY STOP/QUIT 面板）
bash real_rollout.sh ~/lerobot_outputs/act_piper_real/checkpoints/last/pretrained_model
```

**真机采集要点**：
- HOME 位姿 `[0.0, 0.8, -1.2, 0.0, 0.0, 0.0]`，START 会平滑回到 HOME
- 50 Hz 控制、10 Hz 录制；`--stick_scale` 控制移动速度
- 拖拽示教/遥操作时机器人处于重力补偿悬停模式（`MotionCtrl_1(0,0,0x01)`），退出不坠落

---

## 三、目录结构

```
lerobot_setup/
├── piper_sim.py            # MuJoCo 仿真环境（Piper 模型、IK、相机、渲染）
├── sim_collect_piper.py    # 仿真遥操作采集（手柄/键盘）
├── sim_rollout_piper.py    # 仿真部署推理
├── rollout_piper.sh        # 仿真部署启动脚本
├── sim_demo.sh             # LeRobot 通用链路冒烟（pusht/aloha/libero）

├── real_piper.py           # 真机环境封装（piper_sdk V2 + D435i）
├── real_collect.py         # 真机 cv2 控制面板采集（含 IK EMA 平滑）
├── real_rollout.py         # 真机 cv2 控制面板部署（含 50Hz 插值 + EMA）
├── real_rollout.sh         # 真机部署启动脚本

├── train_piper.sh          # 训练 ACT（日志 + W&B）
├── plot_training.py        # 画 loss / l1_loss / kld_loss 曲线
├── postprocess_smooth.py   # 后处理：对 action 做 EMA 平滑生成新数据集

├── common.sh               # 环境清理 + conda 激活
├── config.env              # 真机 CAN/相机配置
├── 00~06_*.sh              # 真机分步脚本（检查/装环境/录制/训练/部署）
└── assets/piper/           # DeepMind agilex_piper MJCF 模型 + 网格
```

---

## 四、关键设计

### 1. 观测管线（必须一致）
相机预览保持 **4:3（480×360）**，再 resize 到 **96×96** 喂给策略。
采集和部署必须走同一条管线，否则策略看到的分布不一致。

### 2. 抖动治理（踩过的坑）
| 问题 | 根因 | 解法 |
|---|---|---|
| 遥操作顿挫 | 控制频率太低 | **50 Hz 控制 + 10 Hz 录制** |
| 录制动作抖动 | IK 零空间冗余漂移（J4/J6） | IK 输出 **EMA 平滑**（`--ik_smooth`）|
| 部署抖动 | 策略本身输出振荡 | 50 Hz 动作插值 + 策略动作 EMA（`--action_smooth`）|

### 3. 数据质量 > loss
最终标准是 **rollout 成功率**，不是 loss。
loss 降了但抓不起来 = 数据质量问题（示范动作本身抖/不一致）。

### 4. 夹爪标定
真机夹爪物理行程约 100 mm。在**闭合位置**设零位，再 `SetSDKGripperRangeParam(0.0, 0.1025)`。

---

## 五、实验结果

| 链路 | 结果 |
|---|---|
| 仿真 sim2sim | **5/5（100%）** 抓取成功，方块抬到 0.29~0.40 m |
| 真机 ACT | 30 条示范，20000 步训练，loss 0.122 / l1_loss 0.087 |

---

## 六、相关仓库

- **piperx-mjlab**：在 MuJoCo Lab（MJLab）里用 **PPO 强化学习**训练 PiPER-X 抓 4cm 方块，
  与这里的模仿学习互补（对比 RL vs IL 的差异）。
- 论文：ACT（arXiv 2304.13705）、Diffusion Policy（2303.04137）、LeRobot（2602.22818）

---

## 七、常见问题

| 现象 | 原因 / 解法 |
|---|---|
| `cublasLtCreate` 崩溃 | 未清 `LD_LIBRARY_PATH`，用 `common.sh`/`activate_env` |
| huggingface proxy 报错 | 未清代理变量，同上 |
| `Output directory already exists` | 用 `--overwrite`（会备份为 `.bak.<时间戳>`，不删数据）|
| rollout 末尾段错误 | MuJoCo GL 退出清理崩溃，结果已打印，不影响 |
| 相机 `Frame didn't arrive` | D435i 掉到 USB 2.1，检查 USB 3.2 连接 |
| gripper 只开 9.7mm | 零位没在闭合位置设，重新标定 |

---

## 八、License

本项目代码用于学习/研究。`assets/piper/` 下的 MJCF 模型来自 DeepMind agilex_piper（Apache-2.0）。
