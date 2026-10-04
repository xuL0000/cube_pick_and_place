#!/usr/bin/env python
"""DAgger 式恢复数据采集：策略先跑，跑偏后 FSM 从当前状态接管完成，
把整段「策略 + 恢复」轨迹录成新 demo，直接教策略从自身误差状态恢复。

原理（DAgger v2，方案 2）：
  1. 策略阶段：ACT 策略按部署配置（joint_angle + temporal ensemble 0.9）直播。
     纯 DAgger 重标注：每帧记录的 action 是无状态专家
     （StatelessExpert）从当前状态计算的 IK 关节目标 + 夹爪命令，而非策略
     自身输出；策略动作仍驱动环境。
  2. 接管：检测到偏离后把 env.action_type 切到 eef_pose，同步 p0/R0 到当前
     tcp，FSM 从当前状态重新规划并完成放置：
       - 立方体落回桌面（曾抬起/夹持后掉落） -> FSM 重新抓取
       - 超过 grasp-deadline 仍从未抬起      -> FSM 重新抓取
       - 夹持中搬运无进展（到盘距离窗口内不减小）-> FSM 从 TRANSPORT 继续（提前接管）
       - 夹持状态下超过 hold-deadline        -> FSM 从 TRANSPORT 继续
       - 策略阶段达到 policy-max-steps       -> 按当前状态接管或判负
  3. 恢复段的「偏离状态 + 专家动作」是扩大策略吸引域的关键数据；策略独立
     完成的回合直接保存（等价普通 demo）。失败回合清空缓存重试。

数据集格式与 cube_collect_data 完全一致，追加进同一 root 后可直接用
cube_train.py 继续训练（无需任何改动）。

Usage:
    python cube_dagger_collect.py --num_demo 20        # 追加到 ./demo_data（默认）
    python cube_dagger_collect.py --num_demo 50 --root ./demo_data_dagger --delete-old
"""
import os
import sys
import argparse
import json
from collections import deque
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image
import torchvision
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps

from mujoco_env.y_env import SimpleEnv
from mujoco_env.ik import solve_ik
from cube_collect_data import (ContactGraspChecker, wait_until_settled,
                               CubePickPlaceFSM, State, build_dataset,
                               CUBE_BODY, PLATE_BODY, PLATE_TOP_SITE)

LIFT_Z_MARGIN = 0.06    # 与部署端一致的抬起判定（cube 相对初始高度抬起 6cm）
HOLD_MIN_GEOMS = 2      # 判定「夹持中」所需的最少夹爪接触 geom 数（与采集端一致）

# 固定配置（不再暴露为命令行参数）
REPO_NAME = "omy_cube_pnp"
TASK_NAME = "Put cube on the plate"
XML_PATH = "./asset/example_scene_cube.xml"
POLICY_MAX_STEPS = 400     # 策略阶段单回合最大步数（20Hz）
FSM_MAX_STEPS = 1200       # FSM 恢复阶段最大步数（超过判负重试）
MAX_RETRIES = 30           # 单回合最大失败重试次数
PROGRESS_WINDOW = 40       # 搬运无进展检测窗口（步）
PROGRESS_EPS = 0.01        # 窗口内到盘距离减小不足该值（米）视为无进展


class PlaceSuccessDetector:
    """三条件成功判定（与 cube_deploy.PlaceSuccessDetector 一致）：
    中心在盘内 + 接触盘面且静止 + 与夹爪分离，连续 hold_ticks 个周期成立。"""

    def __init__(self, checker, hold_ticks: int = 20, settle_dz: float = 0.002):
        self.checker = checker
        self.hold_ticks = hold_ticks
        self.settle_dz = settle_dz
        self.counter = 0
        self._z_hist = []

    def reset(self):
        self.counter = 0
        self._z_hist = []

    def check(self, env) -> bool:
        cube = env.env.get_p_body(CUBE_BODY)
        plate = env.env.get_p_body(PLATE_BODY)
        horiz = float(np.linalg.norm(cube[:2] - plate[:2]))
        on_plate = horiz < self.checker.plate_radius
        touching = len(self.checker.contacting_plate_geoms(env.env)) >= 1
        released = len(self.checker.contacting_gripper_geoms(env.env)) == 0

        self._z_hist.append(cube[2])
        if len(self._z_hist) > 10:
            self._z_hist.pop(0)
        settled = True
        if len(self._z_hist) == 10:
            settled = abs(self._z_hist[-1] - self._z_hist[0]) < self.settle_dz

        if on_plate and touching and released and settled:
            self.counter += 1
        else:
            self.counter = 0
        return self.counter >= self.hold_ticks


def is_holding(env, checker, cube_init_z: float) -> bool:
    """立方体当前是否被夹爪夹住（接触数足够且离桌）。"""
    cube_z = float(env.env.get_p_body(CUBE_BODY)[2])
    n_grip = len(checker.contacting_gripper_geoms(env.env))
    return n_grip >= HOLD_MIN_GEOMS and cube_z > cube_init_z + 0.01


def cube_dropped(env, checker, cube_init_z: float, ever_lifted: bool) -> bool:
    """曾抬起，但现在立方体落回桌面且已与夹爪分离。"""
    if not ever_lifted:
        return False
    cube_z = float(env.env.get_p_body(CUBE_BODY)[2])
    return (cube_z < cube_init_z + 0.01
            and len(checker.contacting_gripper_geoms(env.env)) == 0)


# ===========================================================================
# 纯 DAgger 重标注：无状态专家（方案 2b）
# ===========================================================================
class StatelessExpert:
    """从当前状态计算专家动作（7 维 eef delta + 夹爪命令），无内部状态机。

    与 CubePickPlaceFSM 同一套运动参数，但每步只根据当前状态决定目标：
      - 立方体在桌面上 -> 移向立方体上方悬停点，到位后下降，再到位后闭合
      - 立方体被夹持   -> 先抬到安全高度，再移向放置点上方，再下降放置
      - 立方体已在盘上 -> 张开夹爪并后退抬升
    返回的动作不会被施加到环境（策略仍负责驱动），只用于重标注记录，
    调用只读仿真状态、无副作用。
    """

    def __init__(self, env: SimpleEnv, checker: ContactGraspChecker,
                 pos_step: float = 0.007, hover_height: float = 0.12,
                 grasp_offset_z: float = -0.005, place_dz: float = 0.035,
                 lift_margin: float = 0.10, grip_close: float = 0.65,
                 arrive: float = 0.005):
        self.env = env
        self.checker = checker
        self.pos_step = pos_step
        self.hover_height = hover_height
        self.grasp_offset_z = grasp_offset_z
        self.place_dz = place_dz
        self.lift_margin = lift_margin
        self.grip_close = grip_close
        self.arrive = arrive

    def _move_towards(self, target_xyz, gripper: float):
        tcp = self.env.env.get_p_body('tcp_link')
        delta = target_xyz - tcp
        step = np.clip(delta, -self.pos_step, self.pos_step)
        return np.array([step[0], step[1], step[2], 0.0, 0.0, 0.0, gripper],
                        dtype=np.float32)

    def action(self, cube_init_z: float):
        """返回专家动作（7 维 eef delta + 夹爪命令）。"""
        env = self.env
        cube = env.env.get_p_body(CUBE_BODY)
        plate = env.env.get_p_body(PLATE_BODY)
        tcp = env.env.get_p_body('tcp_link')
        n_grip = len(self.checker.contacting_gripper_geoms(env.env))
        n_plate = len(self.checker.contacting_plate_geoms(env.env))

        holding = n_grip >= 2 and cube[2] > cube_init_z + 0.01
        on_plate = (n_plate >= 1 and n_grip == 0
                    and cube[2] < cube_init_z + 0.03)

        if on_plate:
            # 已放好：张开 + 后退抬升
            target = np.array([tcp[0], tcp[1], max(0.95, tcp[2])])
            return self._move_towards(target, 0.0)

        if holding:
            # 被夹持：先抬到安全高度，再平移至放置点上方，再下降
            place_xyz = np.array(
                [plate[0], plate[1],
                 env.env.get_p_site(PLATE_TOP_SITE)[2] + self.place_dz])
            lift_z = min(cube[2] + self.lift_margin, 0.98)
            horiz = float(np.linalg.norm(tcp[:2] - place_xyz[:2]))
            if tcp[2] < lift_z - self.arrive:
                target = np.array([tcp[0], tcp[1], lift_z])
            elif horiz > self.arrive:
                target = np.array([place_xyz[0], place_xyz[1], lift_z])
            else:
                target = place_xyz
            return self._move_towards(target, self.grip_close)

        # 立方体在桌面上：悬停 -> 下降 -> 闭合
        hover = cube + np.array([0.0, 0.0, self.hover_height])
        grasp = cube + np.array([0.0, 0.0, self.grasp_offset_z])
        horiz = float(np.linalg.norm(tcp[:2] - cube[:2]))
        if horiz > self.arrive:
            return self._move_towards(hover, 0.0)
        if tcp[2] > grasp[2] + self.arrive:
            return self._move_towards(grasp, 0.0)
        return self._move_towards(tcp, self.grip_close)   # 到位：闭合


def main():
    parser = argparse.ArgumentParser(description="DAgger recovery demo collection")
    parser.add_argument("--num_demo", type=int, default=20,
                        help="成功采集的 episode 数")
    parser.add_argument("--seed", type=int, default=-1,
                        help="随机种子，负数（如 -1）则随机布局")
    parser.add_argument("--root", type=str, default="./demo_data",
                        help="数据集 root；已存在则追加，--delete-old 可清空重建")
    parser.add_argument("--ckpt", type=str, default="./ckpt/act_cube")
    parser.add_argument("--stats_root", type=str, default="./demo_data",
                        help="策略归一化统计来源（= 训练该 ckpt 的数据集 root）")
    parser.add_argument("--delete-old", action="store_true",
                        help="root 已存在时直接删除，不再交互询问")
    # 接管触发阈值
    parser.add_argument("--grasp_deadline", type=int, default=120,
                        help="超过该步数仍未抬起立方体则接管")
    parser.add_argument("--hold_deadline", type=int, default=100,
                        help="夹持状态下超过该步数未完成放置则接管（提前接管："
                             "趁漂移未饱和就交回 FSM）")
    args = parser.parse_args()

    seed = args.seed if args.seed >= 0 else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, seed={seed}, num_demo={args.num_demo}, "
          f"root={args.root}, ckpt={args.ckpt}")

    # ------------------------------------------------------------- policy
    # 与 cube_deploy 一致的部署配置（temporal ensemble 0.9）；chunk_size 从
    # checkpoint 读取（新旧模型 chunk_size 可能不同，硬编码会导致权重形状不匹配）
    dataset_metadata = LeRobotDatasetMetadata(REPO_NAME, root=args.stats_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: ft for k, ft in features.items()
                       if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items()
                      if k not in output_features}
    with open(Path(args.ckpt) / "config.json") as f:
        ckpt_cfg = json.load(f)
    use_wrist = "observation.wrist_image" in ckpt_cfg["input_features"]
    if not use_wrist:
        input_features.pop("observation.wrist_image", None)

    cfg = ACTConfig(input_features=input_features,
                    output_features=output_features,
                    chunk_size=int(ckpt_cfg["chunk_size"]),
                    n_action_steps=1,
                    temporal_ensemble_coeff=0.9)
    resolve_delta_timestamps(cfg, dataset_metadata)
    policy = ACTPolicy.from_pretrained(args.ckpt, config=cfg,
                                       dataset_stats=dataset_metadata.stats)
    policy.to(device)
    policy.eval()

    # ------------------------------------------------------------ dataset
    dataset = build_dataset(REPO_NAME, args.root, args.delete_old)
    n_episodes_before = len(dataset.meta.episodes)

    # --------------------------------------------------------------- env
    env = SimpleEnv(XML_PATH, action_type='joint_angle', seed=seed)
    checker = ContactGraspChecker(env.env, min_geoms=HOLD_MIN_GEOMS)
    fsm = CubePickPlaceFSM(env, checker)
    expert = StatelessExpert(env, checker)
    success_detector = PlaceSuccessDetector(checker)
    img_transform = torchvision.transforms.ToTensor()

    # ------------------------------------------------------------- utils
    def record_frame():
        """观测在动作施加前读取（与 cube_collect_data 一致）。"""
        ee_pose = env.get_ee_pose()
        agent_img, wrist_img = env.grab_image()
        agent_img = np.array(Image.fromarray(agent_img).resize((256, 256)))
        wrist_img = np.array(Image.fromarray(wrist_img).resize((256, 256)))
        return ee_pose, agent_img, wrist_img

    def add_frame(ee_pose, agent_img, wrist_img, action, recorded=None):
        """记录动作。recorded=None 时取 env.compute_q（真实下发的关节目标）+
        action[6]（夹爪命令）；纯 DAgger 重标注时传入预计算的 recorded。"""
        if recorded is None:
            recorded = np.concatenate(
                [env.compute_q.astype(np.float32),
                 np.array([action[6]], dtype=np.float32)])
        dataset.add_frame({
            "observation.image": agent_img,
            "observation.wrist_image": wrist_img,
            "observation.state": ee_pose,
            "action": recorded,
            "obj_init": env.obj_init_pose,
        }, task=TASK_NAME)

    def takeover(held: bool, reason: str):
        """切换 action_type 并把 FSM 从当前状态切入。"""
        env.action_type = 'eef_pose'
        p, R = env.env.get_pR_body('tcp_link')
        env.p0, env.R0 = p.copy(), R.copy()
        fsm.begin_episode(already_held=held)
        print(f"[dagger] TAKEOVER step={step} reason='{reason}' "
              f"held={held} tcp=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})")

    def reset_episode():
        env.reset(seed=seed)
        wait_until_settled(env)
        policy.reset()
        success_detector.reset()
        env.action_type = 'joint_angle'
        return float(env.env.get_p_body(CUBE_BODY)[2])

    # ------------------------------------------------------------- loop
    n_saved = 0
    n_policy_only = 0
    n_takeover_saved = 0
    attempt = 0
    step = 0
    ever_lifted = False
    phase = 'policy'          # 'policy' -> 策略直播；'fsm' -> FSM 接管恢复
    fsm_steps = 0
    takeovers = []            # (step, reason) 每回合接管记录
    hold_prog = deque()       # 夹持中 (step, 到盘距离) 窗口，用于搬运无进展检测

    cube_init_z = reset_episode()
    print(f"dataset episodes before: {n_episodes_before}")

    while env.env.is_viewer_alive() and n_saved < args.num_demo:
        env.step_env()
        if not env.env.loop_every(HZ=20):
            continue

        if phase == 'policy':
            ee_pose, agent_img, wrist_img = record_frame()
            image = img_transform(Image.fromarray(agent_img))
            data = {
                'observation.state': torch.from_numpy(np.array([ee_pose])).to(device),
                'observation.image': image.unsqueeze(0).to(device),
                'task': [TASK_NAME],
                'timestamp': torch.tensor([step / 20.0]).to(device),
            }
            if use_wrist:
                wrist = img_transform(Image.fromarray(wrist_img))
                data['observation.wrist_image'] = wrist.unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy.select_action(data)
            action = action[0].cpu().detach().numpy()

            # 纯 DAgger 重标注（方案 2b）：记录的 action 是无状态专家基于
            # 当前状态给出的 IK 关节目标 + 夹爪命令；策略动作仍驱动环境
            exp_action = expert.action(cube_init_z)
            p, R = env.env.get_pR_body('tcp_link')
            q_ik, _, _ = solve_ik(
                env=env.env, joint_names_for_ik=env.joint_names,
                body_name_trgt='tcp_link',
                q_init=env.env.get_qpos_joints(joint_names=env.joint_names),
                p_trgt=p + exp_action[:3], R_trgt=R,
                max_ik_tick=50, ik_stepsize=1.0, ik_eps=1e-2,
                ik_th=np.radians(5.0), render=False,
                verbose_warning=False)
            recorded = np.concatenate(
                [q_ik.astype(np.float32),
                 np.array([exp_action[6]], dtype=np.float32)])
            env.step(action)
            add_frame(ee_pose, agent_img, wrist_img, action, recorded)
            env.render()
            step += 1

            cube_z = float(env.env.get_p_body(CUBE_BODY)[2])
            if not ever_lifted and cube_z > cube_init_z + LIFT_Z_MARGIN:
                ever_lifted = True

            # 策略独立成功 -> 直接保存（等价普通 demo）
            if success_detector.check(env):
                dataset.save_episode()
                n_saved += 1
                n_policy_only += 1
                print(f">>> Episode {n_saved}/{args.num_demo} saved "
                      f"[policy only, {step} frames]")
                step, ever_lifted, attempt = 0, False, 0
                hold_prog.clear()
                cube_init_z = reset_episode()
                continue

            # 偏离检测 -> 接管（含方案 2a 的提前接管触发）
            held = is_holding(env, checker, cube_init_z)
            reason = None
            if cube_dropped(env, checker, cube_init_z, ever_lifted):
                reason = 'cube dropped'
            elif step >= args.grasp_deadline and not ever_lifted:
                reason = 'no grasp'
            elif held:
                # 夹持中：先看搬运是否无进展（提前接管的核心），再看绝对超时
                dist = float(np.linalg.norm(
                    env.env.get_p_body(CUBE_BODY)[:2] -
                    env.env.get_p_body(PLATE_BODY)[:2]))
                hold_prog.append((step, dist))
                while hold_prog and hold_prog[0][0] < step - PROGRESS_WINDOW:
                    hold_prog.popleft()
                if (hold_prog[-1][0] - hold_prog[0][0] >= PROGRESS_WINDOW
                        and hold_prog[-1][1]
                        >= hold_prog[0][1] - PROGRESS_EPS
                        and hold_prog[-1][1]
                        > checker.plate_radius + 0.02):
                    reason = 'transport no progress'
                elif step >= args.hold_deadline:
                    reason = 'hold stall'
            else:
                hold_prog.clear()
            if step >= POLICY_MAX_STEPS and reason is None:
                reason = 'policy timeout (held)' if held else \
                         'policy timeout (no grasp)'
            if reason:
                takeovers.append((step, reason))
                phase = 'fsm'
                fsm_steps = 0
                takeover(held, reason)
            continue

        # -------------------------------------------------- FSM 恢复阶段
        fsm_action, finished = fsm.step()
        ee_pose, agent_img, wrist_img = record_frame()
        env.step(fsm_action)
        add_frame(ee_pose, agent_img, wrist_img, fsm_action)
        env.render()
        step += 1
        fsm_steps += 1

        if finished and fsm.state == State.DONE:
            dataset.save_episode()
            n_saved += 1
            n_takeover_saved += 1
            print(f">>> Episode {n_saved}/{args.num_demo} saved "
                  f"[takeover: {takeovers}, {step} frames]")
            takeovers = []
            step, ever_lifted, attempt = 0, False, 0
            hold_prog.clear()
            phase = 'policy'
            cube_init_z = reset_episode()
        elif finished or fsm_steps >= FSM_MAX_STEPS:
            attempt += 1
            if attempt > MAX_RETRIES:
                print(f"Exceeded {MAX_RETRIES} retries; aborting.")
                break
            print(f"Retry {attempt}/{MAX_RETRIES} "
                  f"(reason: {fsm.failed_reason or 'fsm max steps'})")
            dataset.clear_episode_buffer()
            takeovers = []
            step, ever_lifted = 0, False
            hold_prog.clear()
            phase = 'policy'
            cube_init_z = reset_episode()

    env.env.close_viewer()

    print(f"\nDone. collected {n_saved} dagger episodes into {args.root}")
    print(f"  policy-only : {n_policy_only}")
    print(f"  with takeover: {n_takeover_saved}")


if __name__ == "__main__":
    main()
