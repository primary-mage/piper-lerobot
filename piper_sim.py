#!/usr/bin/env python3
"""松灵 Piper 六轴 + 夹爪的 MuJoCo 仿真环境（基于 DeepMind mujoco_menagerie 的 agilex_piper 模型）。

动作空间：7 维 = 6 个手臂关节(joint1~joint6) + 1 个夹爪(joint7，joint8 镜像联动)。
观测：state = 当前关节角(7维)，image = 顶部相机渲染图。
"""
import os

import mujoco
import numpy as np

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "piper")

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT = "joint7"  # joint8 通过 equality 镜像
GRIPPER_CLOSED = 0.0      # joint7=0   → 两指间距 0（闭合）
GRIPPER_OPEN = 0.035      # joint7=0.035 → 两指间距 ~5cm（张开）
ACTUATOR_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

# 初始关节角（6 手臂 + 夹爪张开）
# 2026-11-12 为「眼在手上相机 + 苹果/香蕉分拣」重新标定：
# 工具轴朝下俯视，红/黄色块上的苹果与香蕉分列画面左右。
# （两块 20cm 色块总跨度 46cm > 相机视野 36cm，无法四角全收；HOME 取两者都可见的平衡点，
#   接近/抓取时手臂移动到各自上方。原值 [0,0.8,-1.2,0,0,0] 是给固定俯视相机的。）
HOME_QPOS = np.array([0.0, 0.6, -1.0, 0.0, 1.2, 0.0, 0.0], dtype=np.float32)

# ===== 相机配置 =====
# 腕部相机外参（相机光学系 → link6），来自真机手眼标定（2026-11-04）。
# 与视觉端 text_grasp.py 的 T_L6_CAM 同源；format：[R|t] 4x4，行主序。
WRIST_CAM_EXTRINSICS = np.array(
    [
        [+0.010210, +0.928004, +0.372430, -0.0730],
        [-0.999915, +0.006436, +0.011376, +0.0027],
        [+0.008160, -0.372514, +0.927991, +0.0557],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# ROS 光学系(x 右, y 下, z 光轴) → MuJoCo 相机系(x 右, y 上, 光轴 -Z)
# 两者差一个绕 X 轴 180° 的旋转。
_OPTICAL_TO_MJ = np.diag([1.0, -1.0, -1.0])

# D435i 彩色：16:9 时垂直 FOV≈42.5°；采集用 640x480(4:3) 时约 55°（水平约 69°）。
# 取 55° 既贴近真机，又能让接近/抓取阶段的方块留在画面内（42.5° 时抓取位会贴边）。
WRIST_FOVY = 55.0
TOP_FOVY = 50.0


def _matrix_to_quat_wxyz(R):
    """3x3 旋转矩阵 → MuJoCo 四元数 (w, x, y, z)。"""
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(R, dtype=np.float64).reshape(9))
    return quat.tolist()

# ===== 场景：水果分拣 =====
TABLE_TOP_Z = 0.08  # 桌面上表面高度（table geom: pos_z=0.04, size_z=0.04）

# 苹果：椭球近似（直径 ~6cm，高 ~5.6cm）
APPLE_XY = (0.30, -0.13)
APPLE_RXY = 0.030
APPLE_RZ = 0.028
APPLE_MASS = 0.150

# 香蕉：沿圆弧排布的胶囊，总长约 14cm，粗 2.6cm
BANANA_XY = (0.30, 0.13)
BANANA_R = 0.013          # 单根胶囊半径（香蕉粗细的一半）
BANANA_SEG_HALF = 0.014   # 胶囊圆柱段半长
BANANA_SEGMENTS = 7
BANANA_RADIUS = 0.115     # 弯曲半径（越小越弯）
BANANA_ARC = 1.30         # 弧张角 (rad)
BANANA_MASS = 0.120

# 色块（20cm × 20cm × 1mm，纯视觉）
MAT_HALF = 0.10
MAT_THICK = 0.0005

OBJECT_NAMES = ("apple", "banana")


def _rot_y(angle):
    """绕 Y 轴旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


# 关节限位（弧度），对应 piper.xml 的 range
JOINT_LIMITS = {
    "joint1": (-2.618, 2.618),
    "joint2": (0.0, 3.14),
    "joint3": (-2.697, 0.0),
    "joint4": (-1.832, 1.832),
    "joint5": (-1.22, 1.22),
    "joint6": (-3.14, 3.14),
    "gripper": (0.0, 0.035),  # 0=张开, 0.035=闭合
}

# 手臂关节限位（弧度），顺序 = joint1..joint6，供 IK 用
ARM_LIMITS = [
    (-2.618, 2.618), (0.0, 3.14), (-2.697, 0.0),
    (-1.832, 1.832), (-1.22, 1.22), (-3.14, 3.14),
]


class PiperEnv:
    """Piper 抓取仿真环境。"""

    def __init__(self, image_size=96, render_width=480, render_height=480, camera="wrist"):
        self.image_size = image_size
        self.render_width = render_width
        self.render_height = render_height
        self.camera = camera

        spec = mujoco.MjSpec.from_file(os.path.join(ASSET_DIR, "piper.xml"))
        # 地面 + 桌子 + 水果(苹果/香蕉) + 色块 + 相机
        spec.worldbody.add_geom(
            name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05], rgba=[0.2, 0.3, 0.4, 1],
        )
        spec.worldbody.add_geom(
            name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.35, 0.30, 0.04], pos=[0.30, 0, 0.04], rgba=[0.75, 0.75, 0.75, 1],
        )

        # --- 苹果（真实曲面网格：视觉 + 凸包碰撞）---
        spec.add_mesh(name="apple_mesh", file="fruits/apple.obj")
        apple = spec.worldbody.add_body(name="apple", pos=[APPLE_XY[0], APPLE_XY[1], TABLE_TOP_Z + APPLE_RZ])
        apple.add_freejoint(name="apple_joint")
        g_apple = apple.add_geom(
            name="apple_geom", type=mujoco.mjtGeom.mjGEOM_MESH,
            mass=APPLE_MASS, rgba=[0.80, 0.10, 0.10, 1.0],
            friction=[1.2, 0.1, 0.01],
        )
        g_apple.meshname = "apple_mesh"
        apple.add_geom(  # 果梗（纯视觉，不参与碰撞）
            name="apple_stem", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.004, 0.012], pos=[0, 0, APPLE_RZ + 0.002],
            rgba=[0.30, 0.20, 0.08, 1.0], mass=0,
            contype=0, conaffinity=0,
        )

        # --- 香蕉（真实曲面网格做视觉 + 胶囊组做碰撞；非凸体网格碰撞不可靠）---
        spec.add_mesh(name="banana_mesh", file="fruits/banana.obj")
        banana = spec.worldbody.add_body(name="banana", pos=[BANANA_XY[0], BANANA_XY[1], TABLE_TOP_Z + BANANA_R])
        banana.add_freejoint(name="banana_joint")
        g_banana = banana.add_geom(  # 视觉网格
            name="banana_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
            mass=0, rgba=[0.95, 0.85, 0.15, 1.0],
            contype=0, conaffinity=0,
        )
        g_banana.meshname = "banana_mesh"
        for i in range(BANANA_SEGMENTS):
            frac = 0.5 if BANANA_SEGMENTS == 1 else i / (BANANA_SEGMENTS - 1)
            ang = (frac - 0.5) * BANANA_ARC
            seg_x = BANANA_RADIUS * np.sin(ang)
            seg_z = BANANA_RADIUS * (1.0 - np.cos(ang))
            banana.add_geom(
                name=f"banana_seg{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[BANANA_R, BANANA_SEG_HALF], pos=[seg_x, 0.0, seg_z],
                quat=_matrix_to_quat_wxyz(_rot_y(ang)),
                mass=BANANA_MASS / BANANA_SEGMENTS,
                rgba=[0.95, 0.85, 0.15, 0.35],  # 碰撞体半透明（调试可见）
                friction=[1.2, 0.1, 0.01],
            )

        # --- 20cm x 20cm 色块（纯视觉，不参与碰撞）---
        # 色块用浅色底（物体鲜艳、底浅），避免和水果本身在视觉上混淆
        for name, xy, rgba in (
            ("mat_apple", APPLE_XY, [0.88, 0.58, 0.58, 0.70]),
            ("mat_banana", BANANA_XY, [0.92, 0.88, 0.55, 0.70]),
        ):
            spec.worldbody.add_geom(
                name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[MAT_HALF, MAT_HALF, MAT_THICK],
                pos=[xy[0], xy[1], TABLE_TOP_Z + MAT_THICK],
                rgba=rgba, contype=0, conaffinity=0, mass=0,
            )

        # 相机：wrist = 眼在手上（挂在 link6），top = 固定俯视（eye-to-hand）
        if camera == "wrist":
            R_l6_mj = WRIST_CAM_EXTRINSICS[:3, :3] @ _OPTICAL_TO_MJ
            t_l6 = WRIST_CAM_EXTRINSICS[:3, 3]
            quat = _matrix_to_quat_wxyz(R_l6_mj)
            spec.body("link6").add_camera(
                name="wrist", pos=t_l6.tolist(), quat=quat, fovy=WRIST_FOVY
            )
        else:
            spec.worldbody.add_camera(
                name="top", pos=[0.30, 0, 1.0], quat=[1, 0, 0, 0], fovy=TOP_FOVY,
            )
        # 目标位置标记（绿色小球，mocap，显示 IK 目标点）
        marker = spec.worldbody.add_body(name="target_marker", pos=[0.30, 0.05, 0.20])
        marker.mocap = True
        marker.add_geom(
            name="target_marker_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.012], rgba=[0.0, 1.0, 0.0, 0.7], contype=0, conaffinity=0,
        )

        self.spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, render_height, render_width)

        self.arm_qpos_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        self.gripper_qpos_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)
        self.act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in ACTUATOR_NAMES]
        self.obj_joint_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
            for name in OBJECT_NAMES
        }
        # qpos 地址 != joint id！多 freejoint 时必须用 jnt_qposadr
        self.obj_qpos_adr = {
            name: int(self.model.jnt_qposadr[jid]) for name, jid in self.obj_joint_ids.items()
        }
        self.obj_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in OBJECT_NAMES
        }
        # 兼容旧接口：默认操作目标 = 苹果
        self.active_object = "apple"
        # 夹爪两指 body id：IK 参考点用「两指中点」——开合对称，中点稳定不随夹爪动
        self.grasp_body_id7 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        self.grasp_body_id8 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link8")
        self.target_mocap_id = 0  # 唯一的 mocap 体
        self.marker_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_marker_geom")

    @property
    def action_dim(self):
        return 7

    def reset(self, seed=0):
        """重置：手臂回 home，苹果/香蕉随机落在各自色块上。返回当前关节角(7维)。"""
        mujoco.mj_resetData(self.model, self.data)
        for i, qid in enumerate(self.arm_qpos_ids):
            self.data.qpos[qid] = HOME_QPOS[i]
        self.data.qpos[self.gripper_qpos_id] = GRIPPER_OPEN  # 夹爪张开（0=闭合, 0.035=张开）

        rng = np.random.default_rng(seed)
        # 苹果：色块中心附近小范围随机
        jid = self.obj_qpos_adr["apple"]
        self.data.qpos[jid:jid + 3] = [
            APPLE_XY[0] + rng.uniform(-0.02, 0.02),
            APPLE_XY[1] + rng.uniform(-0.02, 0.02),
            TABLE_TOP_Z + APPLE_RZ,
        ]
        self.data.qpos[jid + 3:jid + 7] = [1, 0, 0, 0]

        # 香蕉：位置小范围随机 + 朝向随机
        jid = self.obj_qpos_adr["banana"]
        yaw = rng.uniform(-0.6, 0.6)
        self.data.qpos[jid:jid + 3] = [
            BANANA_XY[0] + rng.uniform(-0.02, 0.02),
            BANANA_XY[1] + rng.uniform(-0.02, 0.02),
            TABLE_TOP_Z + BANANA_R,
        ]
        self.data.qpos[jid + 3:jid + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.active_object = "apple"

        mujoco.mj_forward(self.model, self.data)
        return self.get_qpos()

    def get_qpos(self):
        """当前关节角：6 手臂 + 1 夹爪，float32 (7,)。"""
        q = [self.data.qpos[i] for i in self.arm_qpos_ids] + [self.data.qpos[self.gripper_qpos_id]]
        return np.asarray(q, dtype=np.float32)

    def step(self, action, substeps=1):
        """施加 7 维目标关节角，推进 substeps 帧物理。返回 (新关节角, 奖励)。"""
        action = np.asarray(action, dtype=np.float64)
        for i, aid in enumerate(self.act_ids):
            self.data.ctrl[aid] = float(action[i])
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)

        qpos = self.get_qpos()
        # 奖励：目标物体被抬得越高越好（>0.16 视为抓到并抬起）
        obj_z = self.object_height()
        rest_z = TABLE_TOP_Z + min(APPLE_RZ, BANANA_R)
        reward = float(np.clip((obj_z - rest_z) / 0.06, 0.0, 1.0))
        return qpos, reward

    # --- 物体查询（多物体）---
    def object_position(self, name=None):
        """物体位置 (3,)。name 默认当前操作目标。"""
        name = name or self.active_object
        adr = self.obj_qpos_adr[name]
        return self.data.qpos[adr:adr + 3].copy()

    def object_height(self, name=None):
        """物体高度 z。name 默认当前操作目标。"""
        name = name or self.active_object
        return float(self.data.qpos[self.obj_qpos_adr[name] + 2])

    def object_heights(self):
        """所有物体高度 {name: z}，便于调试/判定。"""
        return {n: self.object_height(n) for n in OBJECT_NAMES}

    def max_object_height(self):
        """所有物体中最高的高度（兼容"是否有物体被抓起来"的判定）。"""
        return max(self.object_heights().values())

    # --- 兼容旧接口（采集/部署脚本用 cube_*）---
    def cube_position(self):
        return self.object_position()

    def cube_height(self):
        """当前操作目标的高度（旧接口：原为方块高度）。"""
        return self.object_height()

    def set_active_object(self, name):
        """切换当前操作目标（apple / banana）。"""
        if name not in self.obj_joint_ids:
            raise ValueError(f"unknown object: {name}, choose from {OBJECT_NAMES}")
        self.active_object = name

    def render_image(self, size=None):
        """顶部相机渲染图（RGB uint8）。size 指定输出边长，默认 image_size（数据集用）。

        渲染时临时隐藏绿色目标标记球（alpha=0），避免它污染观测图。
        """
        rgba = self.model.geom_rgba[self.marker_geom_id].copy()
        self.model.geom_rgba[self.marker_geom_id, 3] = 0.0  # 隐藏
        self.renderer.update_scene(self.data, camera=self.camera)
        img = self.renderer.render()  # HWC RGB uint8，原生 render_width×render_height
        self.model.geom_rgba[self.marker_geom_id] = rgba  # 恢复
        size = size or self.image_size
        if img.shape[0] != size:
            import cv2
            img = cv2.resize(img, (size, size))
        return img

    def get_grasp_pos(self):
        """当前夹爪参考点（两指中点）的世界坐标 (3,)。中点不随夹爪开合变化。"""
        return (self.data.xpos[self.grasp_body_id7] + self.data.xpos[self.grasp_body_id8]) / 2.0

    def set_target_marker(self, pos):
        """移动绿色目标标记球到 pos。"""
        self.data.mocap_pos[self.target_mocap_id] = pos

    def solve_ik(self, target_pos, max_iter=60, tol=1e-3, damping=1e-6, null_gain=0.2):
        """阻尼最小二乘 IK + 零空间镇定：求使夹爪中点到达 target_pos 的 6 个手臂关节角。

        零空间镇定：6 关节解 3 维位置是冗余的，DLS 会在冗余自由度间漂移导致关节抖动。
        加一项"拉向当前关节姿态"的零空间任务，让 IK 优先保持关节不动（消抖）。

        不改动真实物理状态（迭代用临时 qpos，结束后恢复）。
        """
        orig = np.array([self.data.qpos[i] for i in self.arm_qpos_ids])
        qpos = orig.copy()
        q_pref = orig.copy()  # 偏好配置 = 当前关节角（零空间拉向这里）
        jacp7 = np.zeros((3, self.model.nv))
        jacr7 = np.zeros((3, self.model.nv))
        jacp8 = np.zeros((3, self.model.nv))
        jacr8 = np.zeros((3, self.model.nv))
        for _ in range(max_iter):
            for i, qid in enumerate(self.arm_qpos_ids):
                self.data.qpos[qid] = qpos[i]
            mujoco.mj_forward(self.model, self.data)
            # 两指中点（开合对称，中点稳定）
            cur = (self.data.xpos[self.grasp_body_id7] + self.data.xpos[self.grasp_body_id8]) / 2.0
            err = np.asarray(target_pos, dtype=np.float64) - cur
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacBody(self.model, self.data, jacp7, jacr7, self.grasp_body_id7)
            mujoco.mj_jacBody(self.model, self.data, jacp8, jacr8, self.grasp_body_id8)
            J = (jacp7[:, :6] + jacp8[:, :6]) / 2.0  # 中点位置雅可比（6 个手臂关节）
            # 主任务：DLS 到达目标
            A = J @ J.T + damping * np.eye(3)
            dq_task = J.T @ np.linalg.solve(A, err)
            # 零空间任务：拉向偏好配置（保持关节不动，消除冗余漂移）
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3))  # (6,3) 伪逆
            N = np.eye(6) - J_pinv @ J  # (6,6) 零空间投影
            dq_null = null_gain * (N @ (q_pref - qpos))
            qpos += dq_task + dq_null
            for i in range(6):
                lo, hi = ARM_LIMITS[i]
                qpos[i] = float(np.clip(qpos[i], lo, hi))
        # 恢复真实物理状态
        for i, qid in enumerate(self.arm_qpos_ids):
            self.data.qpos[qid] = orig[i]
        mujoco.mj_forward(self.model, self.data)
        return qpos.astype(np.float32)

    def close(self):
        self.renderer.close()


if __name__ == "__main__":
    env = PiperEnv()
    q = env.reset(seed=0)
    print("初始关节角:", np.round(q, 3))
    for _ in range(10):
        q, r = env.step(q)
    print("步进 10 步后关节角:", np.round(q, 3), "方块高度:", round(env.cube_height(), 3))
    img = env.render_image()
    print("渲染图 shape:", img.shape, img.dtype)
    env.close()
    print("PiperEnv 自检通过 ✓")
