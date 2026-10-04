#!/usr/bin/env python
"""5cm 立方体 pick-and-place 自动演示采集（接触检测版，精简核心功能）。

与根目录 1.collect_data_auto.py 的区别：
  1. 场景为隔离的 cube_pnp/asset/example_scene_cube.xml（5cm 立方体 + 软垫夹爪）。
  2. 抓取成功判定：闭合阶段实时检测夹爪 geom（含软垫）与立方体的接触数，
     接触 geom 数 >= 2 才认为两侧夹住；自适应地增大闭合命令，接触后再补
     0.15 夹紧量（避免过压挤飞 / 欠压滑脱）。
  3. LIFT 阶段双重确认：接触已触发 + 立方体被带高 > 6cm。
  4. place 成功判定（替代 env.check_success）三条件同时满足：
       a. 立方体中心 xy 处于盘子水平半径内；
       b. 立方体与盘子 geom 存在接触且已静止（不在空中）；
       c. 立方体与夹爪 geom 接触数为 0（已分离）。

action 记录为真实下发的 6 关节 IK 目标 compute_q + 夹爪命令（绝对关节角），
数据结构与原采集脚本完全一致，可直接复用原训练脚本。
失败回合丢弃重试，只保存成功轨迹；数据集已存在时追加续写（--delete-old 重建）。

Usage:
    python cube_collect_data.py                        # 30 demos, ./demo_data
    python cube_collect_data.py --num_demo 100 --seed -1
    python cube_collect_data.py --seed -1 --root ./demo_data_new --delete-old

    # 立方体微小变化：固定名义位置（--seed 0）+ 每回合 ±2cm 扰动
    python cube_collect_data.py --num_demo 100 --seed 0 --cube_jitter 0.02 \
        --root ./demo_data_jitter --delete-old
"""
import os
import sys
import shutil
import argparse
from enum import Enum, auto
from pathlib import Path

# 无论从哪个目录启动，相对路径都以 cube_pnp/ 为准
os.chdir(Path(__file__).resolve().parent)
# 复用项目根目录的 mujoco_env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import mujoco
from PIL import Image
from mujoco_env.y_env import SimpleEnv
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# ===========================================================================
# Configuration（固定配置，不再暴露为命令行参数）
# ===========================================================================
XML_PATH = "./asset/example_scene_cube.xml"
REPO_NAME = "omy_cube_pnp"
TASK_NAME = "Put cube on the plate"
MIN_CONTACT_GEOMS = 2    # 判定夹住所需的最少接触夹爪 geom 数
MAX_RETRIES = 30         # 单个 episode 的最大失败重试次数


class State(Enum):
    """立方体 pick-and-place 状态机。"""
    APPROACH = auto()         # 移动到立方体上方悬停点
    DESCEND = auto()          # 垂直下降到抓取高度
    CLOSE = auto()            # 自适应闭合：接触触发 + 补夹紧量
    LIFT = auto()             # 垂直抓起（接触 + 抬升位移双重确认）
    TRANSPORT = auto()        # 带立方体水平移动到盘子上方
    PLACE_DESCEND = auto()    # 下降到放置高度
    RELEASE = auto()          # 张开夹爪、等待立方体落稳
    RETREAT = auto()          # 抬起离开并做放置三条件判定
    DONE = auto()             # 本回合成功
    FAIL = auto()             # 本回合失败，丢弃数据重试


# 数据特征（与 1.collect_data_auto.py 完全一致）
DATASET_FEATURES = {
    "observation.image": {
        "dtype": "image", "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.wrist_image": {
        "dtype": "image", "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32", "shape": (6,),
        "names": ["state"],  # x, y, z, roll, pitch, yaw
    },
    "action": {
        "dtype": "float32", "shape": (7,),
        "names": ["action"],  # 6 joint angles + 1 gripper
    },
    "obj_init": {
        "dtype": "float32", "shape": (6,),
        "names": ["obj_init"],
    },
}

CUBE_BODY = 'body_obj_mug_5'          # 保留 mug 命名以兼容数据集接口
PLATE_BODY = 'body_obj_plate_11'
CUBE_GEOM = 'cube_5cm_geom'
PLATE_RADIUS_SITE = 'horizontal_radius_site_plate_11'
PLATE_TOP_SITE = 'top_site_plate_11'
GRIPPER_BODIES = ['rh_p12_rn_r1', 'rh_p12_rn_r2', 'rh_p12_rn_l1', 'rh_p12_rn_l2']
GRIPPER_PAD_GEOMS = ['pad_r2', 'pad_l2']


# ===========================================================================
# 接触检测
# ===========================================================================
class ContactGraspChecker:
    """基于 MuJoCo 接触点列表检测夹爪/立方体/盘子三者的接触关系。"""

    def __init__(self, mjc_env, min_geoms: int = 2):
        m = mjc_env.model
        self.m = m
        self.min_geoms = min_geoms
        self.cube_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM)
        assert self.cube_gid >= 0, f'找不到 geom {CUBE_GEOM}'

        # 夹爪 geom 集合：四个指尖 body 下全部 geom + 两个软垫 geom
        self.grip_gids = set()
        for bn in GRIPPER_BODIES:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
            for i in range(m.ngeom):
                if m.geom_bodyid[i] == bid:
                    self.grip_gids.add(i)
        for pn in GRIPPER_PAD_GEOMS:
            pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, pn)
            if pid >= 0:
                self.grip_gids.add(pid)

        # 盘子 geom 集合：盘子 geom 挂在子 body object_plate_11 下，
        # 需收集 body_obj_plate_11 整棵子树的 geom（用 parentid BFS）。
        plate_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PLATE_BODY)
        subtree_body_ids = {plate_bid}
        frontier = [plate_bid]
        while frontier:
            parent = frontier.pop()
            for b in range(m.nbody):
                if m.body_parentid[b] == parent and b not in subtree_body_ids:
                    subtree_body_ids.add(b)
                    frontier.append(b)
        self.plate_gids = set()
        for i in range(m.ngeom):
            if m.geom_bodyid[i] in subtree_body_ids:
                self.plate_gids.add(i)
        assert len(self.plate_gids) > 0, '盘子 geom 集合为空，检查 body 结构'

        # 盘子水平半径：horizontal_radius site 相对盘 body 的水平距离（模型常数）
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, PLATE_RADIUS_SITE)
        self.plate_radius = float(np.linalg.norm(m.site_pos[sid][:2]))

    def _contacts_with(self, mjc_env, target_gid: int, other_gids: set) -> set:
        """返回当前与 target geom 接触的 other geom id 集合。"""
        d = mjc_env.data
        hit = set()
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = c.geom1, c.geom2
            if g1 == target_gid and g2 in other_gids:
                hit.add(g2)
            elif g2 == target_gid and g1 in other_gids:
                hit.add(g1)
        return hit

    def contacting_gripper_geoms(self, mjc_env) -> set:
        return self._contacts_with(mjc_env, self.cube_gid, self.grip_gids)

    def contacting_plate_geoms(self, mjc_env) -> set:
        return self._contacts_with(mjc_env, self.cube_gid, self.plate_gids)

    def grasp_info(self, mjc_env):
        """返回 (是否夹住, 接触夹爪geom数)。"""
        n = len(self.contacting_gripper_geoms(mjc_env))
        return n >= self.min_geoms, n


# ===========================================================================
# 立方体 pick-and-place 状态机
# ===========================================================================
class CubePickPlaceFSM:
    """每个 20Hz 控制周期调用一次 step，返回 (7维 eef delta action, finished)。

    action = [dx, dy, dz, droll, dpitch, dyaw, gripper]，gripper 0=张开 1=闭合。
    """

    def __init__(self, env: SimpleEnv, checker: ContactGraspChecker,
                 pos_step: float = 0.007,        # 每周期最大平移 7mm
                 hover_height: float = 0.12,     # 立方体上方悬停高度
                 grasp_offset_z: float = -0.005, # 抓取点相对立方体中心（软垫正对侧壁中段）
                 lift_height: float = 0.18,      # 抓起后相对初始的抬升量
                 lift_check: float = 0.06,       # 抬起位移成功阈值
                 place_dz: float = 0.035,        # 放置 tcp 距盘面的高度（释放后自然落盘）
                 close_start: float = 0.5,       # 自适应闭合起始命令
                 close_step: float = 0.05,       # 闭合命令步进
                 close_ramp_ticks: int = 8,      # 每隔多少 tick 步进一次（给物理稳定时间）
                 close_squeeze: float = 0.15,    # 接触触发后的额外夹紧量
                 close_hold_ticks: int = 300,    # CLOSE 状态总超时
                 release_ticks: int = 30,        # 张爪等待落稳
                 verify_ticks: int = 20,         # 抬升到位后等待沉降再判定
                 settle_dz: float = 0.002,       # 判定静止时 cube z 的 10-tick 变化阈值
                 max_ticks_per_state: int = 300):
        self.env = env
        self.checker = checker
        self.pos_step = pos_step
        self.hover_height = hover_height
        self.grasp_offset_z = grasp_offset_z
        self.lift_height = lift_height
        self.lift_check = lift_check
        self.place_dz = place_dz
        self.close_start = close_start
        self.close_step = close_step
        self.close_ramp_ticks = close_ramp_ticks
        self.close_squeeze = close_squeeze
        self.close_hold_ticks = close_hold_ticks
        self.release_ticks = release_ticks
        self.verify_ticks = verify_ticks
        self.settle_dz = settle_dz
        self.max_ticks_per_state = max_ticks_per_state

        self.state = None
        self.ticks_in_state = 0
        self.failed_reason = None

        # 本回合目标
        self.cube0 = None
        self.grasp_xyz = None
        self.hover_xyz = None
        self.lift_z = None
        self.place_xyz = None
        # 自适应闭合状态
        self.grip_cmd = None
        self.grip_close = None
        self.grasped_flag = False
        # RETREAT 到位时刻（到位后开始 verify 等待）与 cube z 历史
        self._arrive_tick = None
        self._cube_z_hist = []

    # ------------------------------------------------------------------ utils
    def _tcp_xyz(self):
        return self.env.env.get_p_body('tcp_link')

    def _cube_xyz(self):
        return self.env.env.get_p_body(CUBE_BODY)

    def _plate_xyz(self):
        return self.env.env.get_p_body(PLATE_BODY)

    def _plate_top_z(self):
        return self.env.env.get_p_site(PLATE_TOP_SITE)[2]

    def _transition(self, new_state: State):
        self.state = new_state
        self.ticks_in_state = 0

    def _timed_out(self):
        return self.ticks_in_state >= self.max_ticks_per_state

    def _fail(self, reason: str):
        self.failed_reason = reason
        print(f"[FSM] FAIL: {reason}")
        self._transition(State.FAIL)

    def _move_towards(self, target_xyz, gripper: float,
                      arrive: float = 0.005):
        tcp = self._tcp_xyz()
        delta = target_xyz - tcp
        step = np.clip(delta, -self.pos_step, self.pos_step)
        action = np.array([step[0], step[1], step[2], 0.0, 0.0, 0.0, gripper],
                          dtype=np.float32)
        return action, np.linalg.norm(delta) < arrive

    def _wait_action(self, gripper: float):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper], dtype=np.float32)

    # ----------------------------------------------------------------- public
    def begin_episode(self, already_held: bool = False):
        """环境 reset 且立方体落稳后调用，读取物体位姿并计算本回合目标。

        already_held=True 用于 DAgger 接管：立方体已在夹爪中，跳过抓取阶段，
        从当前状态直接进入 TRANSPORT 完成放置。
        """
        self.state = None
        self.ticks_in_state = 0
        self.failed_reason = None
        self.cube0 = None
        self.grip_cmd = None
        self.grip_close = None
        self.grasped_flag = False
        self._arrive_tick = None
        self._cube_z_hist = []

        cube = self._cube_xyz()
        plate = self._plate_xyz()
        self.cube0 = cube.copy()
        self.place_xyz = np.array(
            [plate[0], plate[1], self._plate_top_z() + self.place_dz])
        if already_held:
            # 抬到当前夹持高度上方 10cm（不超过 0.98，避免超出工作空间）
            self.lift_z = min(cube[2] + 0.10, 0.98)
            self.grip_cmd = 0.65
            self.grip_close = 0.65
            self.grasped_flag = True
            self._transition(State.TRANSPORT)
        else:
            self.grasp_xyz = cube + np.array([0.0, 0.0, self.grasp_offset_z])
            self.hover_xyz = cube + np.array([0.0, 0.0, self.hover_height])
            self.lift_z = cube[2] + self.lift_height
            self._transition(State.APPROACH)

    def step(self):
        """返回 (action, finished)；finished 时状态为 DONE 或 FAIL。"""
        self.ticks_in_state += 1
        s = self.state

        if s == State.APPROACH:
            action, arrived = self._move_towards(self.hover_xyz, 0.0)
            if arrived:
                self._transition(State.DESCEND)
            elif self._timed_out():
                self._fail("approach timeout")
            return action, False

        if s == State.DESCEND:
            action, arrived = self._move_towards(self.grasp_xyz, 0.0, arrive=0.004)
            if arrived:
                self._transition(State.CLOSE)
            elif self._timed_out():
                self._fail("descend timeout")
            return action, False

        if s == State.CLOSE:
            # 自适应接触闭合：命令从 0.5 起每 8 tick +0.05；
            # 一旦两侧接触（>=min_geoms）立刻在当前命令上 +0.15 夹紧并进入 LIFT。
            if self.grip_cmd is None:
                self.grip_cmd = self.close_start
            grasped, ncon = self.checker.grasp_info(self.env.env)
            if grasped:
                self.grip_close = min(self.grip_cmd + self.close_squeeze, 1.0)
                self.grasped_flag = True
                print(f"[FSM] CLOSE 接触geom={ncon} 触发夹紧 "
                      f"grip_close={self.grip_close:.2f}")
                self._transition(State.LIFT)
                return self._wait_action(self.grip_close), False
            if self.ticks_in_state % self.close_ramp_ticks == 0:
                self.grip_cmd = min(self.grip_cmd + self.close_step, 1.0)
            if self.ticks_in_state >= self.close_hold_ticks:
                self._fail(f"close: no enough contact (geom={ncon})")
            return self._wait_action(self.grip_cmd), False

        if s == State.LIFT:
            target = np.array([self.grasp_xyz[0], self.grasp_xyz[1], self.lift_z])
            action, arrived = self._move_towards(target, self.grip_close)
            if arrived:
                cube_z = self._cube_xyz()[2]
                lifted = cube_z > self.cube0[2] + self.lift_check
                _, ncon = self.checker.grasp_info(self.env.env)
                print(f"[FSM] LIFT 完成: 接触geom={ncon} cube_z={cube_z:.3f} "
                      f"(初始 {self.cube0[2]:.3f}) lifted={lifted}")
                if self.grasped_flag and lifted:
                    self._transition(State.TRANSPORT)
                else:
                    self._fail("grasp failed: cube was not lifted")
            elif self._timed_out():
                self._fail("lift timeout")
            return action, False

        if s == State.TRANSPORT:
            target = np.array([self.place_xyz[0], self.place_xyz[1], self.lift_z])
            action, arrived = self._move_towards(target, self.grip_close)
            if arrived:
                self._transition(State.PLACE_DESCEND)
            elif self._timed_out():
                self._fail("transport timeout")
            return action, False

        if s == State.PLACE_DESCEND:
            action, arrived = self._move_towards(self.place_xyz, self.grip_close,
                                                 arrive=0.005)
            if arrived:
                self._transition(State.RELEASE)
            elif self._timed_out():
                self._fail("place descend timeout")
            return action, False

        if s == State.RELEASE:
            if self.ticks_in_state > self.release_ticks:
                self._transition(State.RETREAT)
            return self._wait_action(0.0), False

        if s == State.RETREAT:
            # 先抬到安全高度；到位后每 tick 记录 cube z，等待 verify_ticks
            # 沉降后做放置三条件判定（原 RETREAT + VERIFY 合并）
            tcp = self._tcp_xyz()
            target = np.array([tcp[0], tcp[1], max(0.95, tcp[2])])
            action, arrived = self._move_towards(target, 0.0)
            if not arrived:
                if self._timed_out():
                    self._fail("retreat timeout")
                return action, False
            if self._arrive_tick is None:
                self._arrive_tick = self.ticks_in_state
            self._cube_z_hist.append(self._cube_xyz()[2])
            if self.ticks_in_state - self._arrive_tick >= self.verify_ticks:
                ok, diag = self._check_place_success()
                if ok:
                    self._transition(State.DONE)
                    return self._wait_action(0.0), True
                self._fail(f"place check failed: {diag}")
                return self._wait_action(0.0), True
            return self._wait_action(0.0), False

        # DONE / FAIL
        return self._wait_action(0.0), True

    def _check_place_success(self):
        """place 成功三条件：中心在盘内 + 已接触盘面且静止 + 与夹爪分离。

        cube z 历史由 RETREAT 到位后每 tick 记录（self._cube_z_hist）。
        """
        cube = self._cube_xyz()
        plate = self._plate_xyz()
        horiz = float(np.linalg.norm(cube[:2] - plate[:2]))
        on_plate = horiz < self.checker.plate_radius

        n_plate_contact = len(self.checker.contacting_plate_geoms(self.env.env))
        touching_plate = n_plate_contact >= 1
        settled = True
        if len(self._cube_z_hist) >= 10:
            settled = (abs(self._cube_z_hist[-1] - self._cube_z_hist[-10])
                       < self.settle_dz)

        n_grip = len(self.checker.contacting_gripper_geoms(self.env.env))
        released = n_grip == 0

        ok = on_plate and touching_plate and settled and released
        diag = (f"horiz={horiz:.3f}(<{self.checker.plate_radius:.3f}) "
                f"plate_contact={n_plate_contact} settled={settled} "
                f"gripper_contact={n_grip}")
        return ok, diag


# ===========================================================================
# Dataset helpers
# ===========================================================================
def build_dataset(repo_name: str, root: str, delete_old: bool) -> LeRobotDataset:
    """root 已存在时追加续写（delete_old=True 则先删除重建）。"""
    if os.path.exists(root):
        if delete_old:
            shutil.rmtree(root)
        else:
            print("Load from previous dataset (append mode)")
            return LeRobotDataset(repo_name, root=root)
    return LeRobotDataset.create(
        repo_id=repo_name,
        root=root,
        robot_type="omy",
        fps=20,
        features=DATASET_FEATURES,
        image_writer_threads=10,
        image_writer_processes=5,
    )


def wait_until_settled(env: SimpleEnv, max_ticks: int = 400,
                       window: int = 20, dz: float = 1e-3):
    """reset 后轮询物理 tick，直到立方体与盘子均落稳。

    SimpleEnv.reset 内部只 settle 100 tick；5cm 立方体初始嵌入桌面会被弹起，
    需要约 140 tick 才完全静止，故采集前在此补等。
    """
    def obj_zs():
        return (env.env.get_p_body(CUBE_BODY)[2],
                env.env.get_p_body(PLATE_BODY)[2])

    hist = []
    for _ in range(max_ticks):
        env.step_env()
        env.env.render()
        hist.append(obj_zs())
        if len(hist) > window:
            hist.pop(0)
        if len(hist) == window:
            arr = np.array(hist)
            if np.abs(arr[-1] - arr[0]).max() < dz:
                return
    print("[warn] wait_until_settled 达到最大 tick，物体可能仍未静止")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Auto cube pick-and-place data collection (contact-based)")
    parser.add_argument("--num_demo", type=int, default=30,
                        help="成功采集的 episode 数")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子，传负数（如 -1）则不固定")
    parser.add_argument("--root", type=str, default="./demo_data")
    parser.add_argument("--cube_jitter", type=float, default=0.0,
                        help="每回合立方体在 reset 位置基础上加 ±J 的 x/y 均匀扰动"
                             "（m，0=关闭）；与 --seed 0 组合 = 固定名义位置 + 微小变化")
    parser.add_argument("--delete-old", action="store_true",
                        help="root 已存在时删除重建（否则追加续写）")
    args = parser.parse_args()

    seed = args.seed if args.seed >= 0 else None
    # 独立 RNG：SimpleEnv.reset 内部会 np.random.seed(0) 重置全局 RNG，
    # 立方体扰动改用独立 generator，保证逐回合真实随机（不受 reset 干扰）
    rng = np.random.default_rng(args.seed if args.seed >= 0 else None)

    dataset = build_dataset(REPO_NAME, args.root, args.delete_old)

    env = SimpleEnv(XML_PATH, action_type='eef_pose',
                    state_type='joint_angle', seed=seed)
    checker = ContactGraspChecker(env.env, min_geoms=MIN_CONTACT_GEOMS)
    fsm = CubePickPlaceFSM(env, checker)
    print(f"plate radius={checker.plate_radius:.3f}m, "
          f"gripper geoms={len(checker.grip_gids)}, "
          f"plate geoms={len(checker.plate_gids)}")

    episode_id = 0
    attempt = 0

    def reset_episode():
        env.reset(seed=seed)
        if args.cube_jitter > 0:
            p_cube = env.env.get_p_body(CUBE_BODY).copy()
            p_cube[:2] += rng.uniform(-args.cube_jitter, args.cube_jitter, size=2)
            env.env.set_p_base_body(body_name=CUBE_BODY, p=p_cube)
            env.env.forward(increase_tick=False)
            # obj_init 同步为实际初始位姿（立方体已含扰动）
            mug_p, plate_p = env.get_obj_pose()
            env.obj_init_pose = np.concatenate(
                [mug_p, plate_p]).astype(np.float32)
        wait_until_settled(env)
        fsm.begin_episode()

    try:
        reset_episode()
        while env.env.is_viewer_alive() and episode_id < args.num_demo:
            env.step_env()
            if not env.env.loop_every(HZ=20):
                continue

            action, finished = fsm.step()

            # 观测在动作施加前读取；action 记录真实下发的 compute_q（无滞后一帧）
            ee_pose = env.get_ee_pose()
            agent_image, wrist_image = env.grab_image()
            agent_image = np.array(Image.fromarray(agent_image).resize((256, 256)))
            wrist_image = np.array(Image.fromarray(wrist_image).resize((256, 256)))
            env.step(action)
            action_recorded = np.concatenate(
                [env.compute_q.astype(np.float32),
                 np.array([action[6]], dtype=np.float32)])

            dataset.add_frame({
                "observation.image": agent_image,
                "observation.wrist_image": wrist_image,
                "observation.state": ee_pose,
                "action": action_recorded,
                "obj_init": env.obj_init_pose,
            }, task=TASK_NAME)

            env.render()

            if finished:
                if fsm.state == State.DONE:
                    dataset.save_episode()
                    episode_id += 1
                    print(f">>> Episode {episode_id}/{args.num_demo} saved.")
                    if episode_id < args.num_demo:
                        reset_episode()
                        attempt = 0
                else:
                    attempt += 1
                    if attempt > MAX_RETRIES:
                        print(f"Exceeded {MAX_RETRIES} retries; aborting.")
                        break
                    print(f"Retry {attempt}/{MAX_RETRIES} "
                          f"(reason: {fsm.failed_reason})")
                    dataset.clear_episode_buffer()
                    reset_episode()
    finally:
        env.env.close_viewer()
        # 新版 lerobot 直接把帧写入 parquet，images 目录仅旧版存在
        images_dir = Path(dataset.root) / 'images'
        if images_dir.exists():
            shutil.rmtree(images_dir)

    print(f"Done. Collected {episode_id} episodes into {args.root}")


if __name__ == "__main__":
    main()
