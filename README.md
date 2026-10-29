# 松灵 Piper + LeRobot 自主抓取 · 仿真闭环（sim2sim）

在 **MuJoCo** 里仿真松灵 Piper 六轴机械臂，用**键盘 / Xbox 手柄做笛卡尔遥操作**采集数据，
训练 **ACT** 策略，再回到仿真里**部署抓取**。全程脱离真机，用于把「采集 → 训练 → 部署」整条链路跑通、摸熟。

> 最终目标是把这套流程迁移到真机（piper_sdk + 拖拽示教），仿真版本验证的是软件链路和手感。

---

## 一、环境要求

| 项 | 要求 |
|---|---|
| 系统 | Ubuntu 24.04（x86_64） |
| 显卡 | NVIDIA RTX 5060（8GB），驱动 ≥ 570 |
| 环境管理 | conda（miniconda3） |
| Python | 3.12 |
| conda 环境 | 已建好的 `lerobot`（含 torch cu130 + lerobot 0.6.1 + piper 插件 + mujoco） |

```bash
conda activate lerobot          # 每次进项目先激活
cd /home/mage/piper_ws/lerobot_setup
```

---

## 二、核心 4 步（采集 → 训练 → 部署 → 看曲线）

### 第 1 步：采集数据（遥操作）

```bash
# Xbox 手柄遥操作（推荐，摇杆控制末端）
python sim_collect_piper.py --num_episodes 30 --overwrite --controller gamepad

# 或键盘遥操作
python sim_collect_piper.py --num_episodes 30 --overwrite --controller keyboard
```

**手柄按键**：

| 手柄 | 作用 |
|---|---|
| 左摇杆 前后/左右 | 末端水平移动（x/y） |
| 右摇杆 上下 | 末端升降（z） |
| A 键 | 夹爪开合切换 |
| B 键 | **丢弃本条**（失败数据不保存） |
| Start | 下一条 episode |
| Select | 退出 |

**键盘按键**：`w/s` 前后、`a/d` 左右、`r/f` 升降、`g` 夹爪、`x` 丢弃、`n` 下一条、`q` 退出

> 采集要点：
> - 每条 = 一次完整的「靠近方块 → 夹爪闭合 → 抬起来」，别塞多个动作
> - 抓失败了按 **B（或 x）** 当场丢弃，别留废数据
> - `--num_episodes 30` 是"采满 30 条**好数据**"（丢弃的不占名额）

### 第 2 步：训练 ACT

```bash
bash train_piper.sh 2000      # 先 2000 步快速验证（约 1 分钟）
bash train_piper.sh 20000     # 正式训练（约 9 分钟）
```

- 训练时会**实时打印 loss**，并自动保存日志到 `~/lerobot_outputs/act_piper_keyboard.log`
- 同时开启 **W&B** 实时曲线（浏览器打开终端里的 `Track this run` 链接）
- 找不到 W&B 链接时：`grep "Track this run" ~/lerobot_outputs/act_piper_keyboard.log`

### 第 3 步：部署推理（带图形界面）

```bash
bash rollout_piper.sh
```

会弹**两个窗口**：
1. **MuJoCo 原生 viewer**：自由视角看机械臂（鼠标旋转/缩放）
2. **"Top Camera (policy view)"**：顶部相机俯拍图 = 策略"眼睛"看到的观测

终端打印每条「方块最高高度」，**>0.16 米 = 抓起来了**。

### 第 4 步：画 loss 曲线

```bash
python plot_training.py ~/lerobot_outputs/act_piper_keyboard.log
# → 生成 *_curves.png（loss / l1_loss / kld_loss / lr 四张图）
```

---

## 三、关键指标怎么读

| 指标 | 含义 | 健康表现 |
|---|---|---|
| **l1_loss** | 动作重建误差（**最重要**） | 持续下降，收敛到 0.1 左右 |
| loss | 总损失 | 下降 |
| kld_loss | VAE KL 散度 | 先降后稳 |
| grdn | 梯度范数 | 稳定，不爆 NaN |
| mem_gb | 显存 | 稳定，不 OOM |

> **最终标准是 rollout 成功率，不是 loss 本身。** loss 降了但抓不起来 = 数据质量问题。

---

## 四、版本管理（git）

已用 git 管理，关键节点打了 tag：

```bash
git log --oneline      # 看历史
git tag                # 看标签

# 回退到某个版本
git reset --hard <commit或tag>
```

| 标签 | 内容 |
|---|---|
| v1-baseline | 纯位置 IK 遥操 |
| v2-yaw-test | 位置 + LT/RT 偏航（测试过，不好用已回退） |
| v1.1-no-marker | 隐藏遥操绿球 |

---

## 五、常见问题

- **训练报 `Output directory already exists`** → 脚本已处理（`rm -rf` 输出目录），若手动跑 `lerobot-train` 记得换 `--output_dir`
- **训练报 `cublasLtCreate`** → 脚本已自动清 `LD_LIBRARY_PATH`，手动跑要 `env -u LD_LIBRARY_PATH`
- **huggingface 报 proxy/socks 错误** → 已装 `socksio` + 脚本自动清代理变量
- **rollout 末尾"段错误"** → 结果已全部打印，是 mujoco GL 退出清理的崩溃，不影响结果
- **采集时夹爪惯性带歪手臂** → 已改斜坡开合（15 帧渐变）
- **采集画面里绿球污染观测** → 已在渲染观测图时隐藏（`v1.1-no-marker` 起生效）

---

## 六、目录说明

| 文件 | 作用 |
|---|---|
| `sim_collect_piper.py` | 手柄/键盘遥操作采集数据 |
| `train_piper.sh` | 训练 ACT（含日志 + W&B） |
| `rollout_piper.sh` | 部署推理（图形界面 + 相机视角） |
| `sim_rollout_piper.py` | rollout 核心逻辑（手动归一化加载 ACT） |
| `piper_sim.py` | Piper MuJoCo 仿真环境（含 IK、相机、绿球标记） |
| `plot_training.py` | 画 loss 曲线 |
| `sim_demo.sh` | 纯仿真冒烟（pusht/aloha/libero，验证 LeRobot 通用链路） |
| `assets/piper/` | DeepMind 官方 `agilex_piper` MJCF 模型 |
| `00~06_*.sh`、`config.env`、`common.sh` | 真机 LeRobot 脚本（CAN + piper_sdk，真机阶段用） |

---

## 七、当前成果

- **20000 步 ACT 策略，sim2sim 抓取成功率 5/5（100%）**，方块稳定抬到 0.29~0.40 米
- loss 10.61 → 0.152，l1_loss 0.65 → 0.116

**下一步方向**（二选一）：
1. 加数据多样性（方块随机位置/朝向）再训，测泛化能力
2. 回真机：把 `piper_sim.py` 换成 `piper_sdk` + CAN，遥操换成拖拽示教
