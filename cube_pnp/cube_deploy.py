#!/usr/bin/env python
"""部署训练好的 ACT 策略到 5cm 立方体 pick-and-place 环境（精简评测工具）。

迁移自根目录 4.deploy.py，改动：
  1. 场景为隔离的 cube_pnp/asset/example_scene_cube.xml，策略为 ckpt/act_cube。
  2. 成功判定使用 cube_collect_data 的接触式三条件（中心在盘内 + 接触盘面且
     静止 + 与夹爪分离），替代旧 env.check_success。
  3. reset 后等待立方体落稳再开始 rollout。
  4. 纯评测模式：固定跑 N 个回合（成败均计数），输出成功率与失败分类。

常用指令：
    # 随机物体初始位置，跑 100 回合（验收标准口径）
    python cube_deploy.py --eval 100 --seed -1

    # 固定布局（seed 0）
    python cube_deploy.py --eval 10 --seed 0

    # 切换 checkpoint / 数据集 root
    python cube_deploy.py --eval 100 --seed -1 --ckpt ./ckpt/act_cube_dagger \
        --root ./demo_data

    # 记录成功回合的 7 维 action（6 关节角 rad + 夹爪）到 CSV
    python cube_deploy.py --eval 10 --log_actions ./logs/act_eval10.csv

    # 立方体微小变化评测（与采集端 --cube_jitter 0.02 对应）
    python cube_deploy.py --eval 10 --seed 0 --cube_jitter 0.02 \
        --ckpt ./ckpt/act_cube_jitter

主要参数：
    --eval N             评测模式：固定跑 N 个回合（成败均计数），输出成功率
    --seed SEED          物体位置随机种子，负数（如 -1）为随机
    --ckpt PATH          模型 checkpoint 目录，默认 ./ckpt/act_cube
    --root PATH          数据集 root（用于读取归一化 stats），默认 ./demo_data
    --n_action_steps K   每 K 步重规划一次、执行预测序列前 K 步（chunk_size=20，
                         K=10 即执行一半）；K=1 时使用 temporal ensemble，
                         K>1 自动关闭 ensemble（lerobot 要求两者互斥）
    --cube_jitter J      每回合立方体在 reset 位置基础上加 ±J 的 x/y 均匀扰动
                         （m，0=关闭）；评测应与训练时的采集配置一致
    --log_actions CSV    非空时把成功回合的 action 关节角逐帧记录到 CSV
                         （含 episode/step 列，失败回合丢弃）
"""
import os
import sys
import csv
import json
import argparse
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
from cube_collect_data import (ContactGraspChecker, wait_until_settled,
                               CUBE_BODY, PLATE_BODY)


# ===========================================================================
# 固定配置（不再暴露为命令行参数）
# ===========================================================================
XML_PATH = "./asset/example_scene_cube.xml"
REPO_NAME = "omy_cube_pnp"
TASK_NAME = "Put cube on the plate"
MAX_STEPS = 400          # 单回合最大控制步数（20Hz，400=20s）
TEMPORAL_COEFF = 0.9     # temporal ensemble 系数（实测 0.01 更差，固定 0.9）


# ===========================================================================
# 成功判定（与采集端三条件一致）
# ===========================================================================
class PlaceSuccessDetector:
    """三条件连续 hold_ticks 个控制周期成立才判定成功，避免接触抖动误判。"""

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


# ===========================================================================
# 失败模式分类
# ===========================================================================
LIFT_Z_MARGIN = 0.06   # cube 相对初始高度抬起超过 6cm 视为曾经抓起
TABLE_Z = 0.825        # 立方体在桌面静止时的中心高度（5cm 立方体半高 2.5cm）

def classify_failure(env, checker, cube_init_z: float, ever_lifted: bool) -> str:
    """回合超时未成功时，依据末态与过程标志分类失败原因。

    GRASP_FAIL     从未抓起（闭合后没带起来）
    DROPPED        曾抓起，末态掉回桌面且离盘远（搬运途中脱落）
    PLACE_OFFSET   立方体已在盘面高度但水平偏出盘子半径
    NOT_RELEASED   到达盘位但夹爪仍与立方体接触（没松开/松开失败）
    TIMEOUT_OTHER  其他（卡住、姿态异常等）
    """
    cube = env.env.get_p_body(CUBE_BODY)
    plate = env.env.get_p_body(PLATE_BODY)
    horiz = float(np.linalg.norm(cube[:2] - plate[:2]))
    npc = len(checker.contacting_plate_geoms(env.env))
    ngc = len(checker.contacting_gripper_geoms(env.env))

    if not ever_lifted:
        return "GRASP_FAIL"
    if npc >= 1 and ngc > 0:
        return "NOT_RELEASED"
    if (npc >= 1 or abs(cube[2] - TABLE_Z) < 0.04) and horiz >= checker.plate_radius:
        return "PLACE_OFFSET"
    if cube[2] < TABLE_Z + 0.04 and horiz >= checker.plate_radius:
        return "DROPPED"
    return "TIMEOUT_OTHER"


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Deploy ACT policy on cube env")
    parser.add_argument("--seed", type=int, default=0,
                        help="环境随机种子，传负数（如 -1）则随机位置")
    parser.add_argument("--root", type=str, default="./demo_data")
    parser.add_argument("--ckpt", type=str, default="./ckpt/act_cube")
    parser.add_argument("--eval", type=int, default=10, metavar="N",
                        help="评测模式：固定跑 N 个回合（成败均计数），输出成功率")
    parser.add_argument("--n_action_steps", type=int, default=1,
                        help="每 K 步重规划一次、执行预测序列前 K 步（chunk_size=20，"
                             "K=10 即执行一半）；K>1 时自动关闭 temporal ensemble"
                             "（lerobot 要求两者互斥）")
    parser.add_argument("--cube_jitter", type=float, default=0.0,
                        help="每回合立方体在 reset 位置基础上加 ±J 的 x/y 均匀扰动"
                             "（m，0=关闭）；评测应与训练时的采集配置一致")
    parser.add_argument("--log_actions", type=str, default="", metavar="CSV",
                        help="非空时，把每个成功回合的 7 维 action（6 关节角 rad + "
                             "夹爪指令）逐帧记录到该 CSV（含 episode/step 列，"
                             "失败回合丢弃）")
    args = parser.parse_args()

    seed = args.seed if args.seed >= 0 else None
    # 独立 RNG：SimpleEnv.reset 内部会 np.random.seed(0) 重置全局 RNG，
    # 立方体扰动改用独立 generator，保证逐回合真实随机（不受 reset 干扰）
    rng = np.random.default_rng(args.seed if args.seed >= 0 else None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_target = args.eval
    print(f"device={device}, seed={seed}, ckpt={args.ckpt}, EVAL {n_target} eps")

    # ------------------------------------------------------------- policy
    dataset_metadata = LeRobotDatasetMetadata(REPO_NAME, root=args.root)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: ft for k, ft in features.items()
                       if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items()
                      if k not in output_features}

    # chunk_size 从 checkpoint 读取（必须与训练一致，否则权重形状不匹配）
    with open(Path(args.ckpt) / "config.json") as f:
        ckpt_cfg = json.load(f)
    chunk_size = int(ckpt_cfg["chunk_size"])
    if args.n_action_steps > chunk_size:
        raise SystemExit(f"--n_action_steps {args.n_action_steps} 超过 chunk_size "
                         f"{chunk_size}（模型结构限制）")
    use_ensemble = args.n_action_steps == 1
    cfg = ACTConfig(input_features=input_features,
                    output_features=output_features,
                    chunk_size=chunk_size, n_action_steps=args.n_action_steps,
                    temporal_ensemble_coeff=(
                        TEMPORAL_COEFF if use_ensemble else None))
    print(f"inference: chunk_size={chunk_size}, "
          f"n_action_steps={args.n_action_steps}, "
          f"temporal_ensemble={'on' if use_ensemble else 'off'}")
    resolve_delta_timestamps(cfg, dataset_metadata)
    policy = ACTPolicy.from_pretrained(args.ckpt, config=cfg,
                                       dataset_stats=dataset_metadata.stats)
    policy.to(device)
    policy.eval()

    # --------------------------------------------------------------- env
    env = SimpleEnv(XML_PATH, action_type='joint_angle', seed=seed)
    checker = ContactGraspChecker(env.env, min_geoms=2)
    success_detector = PlaceSuccessDetector(checker)
    img_transform = torchvision.transforms.ToTensor()

    def reset_episode():
        env.reset(seed=seed)
        if args.cube_jitter > 0:
            p_cube = env.env.get_p_body(CUBE_BODY).copy()
            p_cube[:2] += rng.uniform(-args.cube_jitter, args.cube_jitter, size=2)
            env.env.set_p_base_body(body_name=CUBE_BODY, p=p_cube)
            env.env.forward(increase_tick=False)
        wait_until_settled(env)
        policy.reset()
        success_detector.reset()
        return float(env.env.get_p_body(CUBE_BODY)[2])

    # ------------------------------------------------------------ rollout
    step = 0
    n_success = 0
    n_attempt = 0
    fail_categories = {}          # 失败模式 -> 次数
    ever_lifted = False
    action_log = []               # --log_actions：已成功回合的帧（episode=成功序号）
    episode_log = []              # 当前回合缓冲（成功才并入 action_log）
    cube_init_z = reset_episode()
    n_attempt += 1

    while env.env.is_viewer_alive():
        env.step_env()
        if not env.env.loop_every(HZ=20):
            continue

        # 构造观测（双视角 agentview + wrist；+ ee pose，与训练输入一致）
        state = env.get_ee_pose()
        image, wrist_image = env.grab_image()
        image = img_transform(Image.fromarray(image).resize((256, 256)))
        wrist_image = img_transform(
            Image.fromarray(wrist_image).resize((256, 256)))
        data = {
            'observation.state': torch.from_numpy(np.array([state])).to(device),
            'observation.image': image.unsqueeze(0).to(device),
            'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
            'task': [TASK_NAME],
            'timestamp': torch.tensor([step / 20.0]).to(device),
        }

        with torch.no_grad():
            action = policy.select_action(data)
        action = action[0].cpu().detach().numpy()

        if args.log_actions:
            episode_log.append([step, *action.tolist()])
        env.step(action)
        env.render()
        step += 1

        # 过程标志：cube 是否曾被带离初始高度
        if not ever_lifted and \
                float(env.env.get_p_body(CUBE_BODY)[2]) > cube_init_z + LIFT_Z_MARGIN:
            ever_lifted = True

        # 成功判定
        if success_detector.check(env):
            n_success += 1
            cube = env.env.get_p_body(CUBE_BODY)
            print(f"[eval {n_attempt}/{n_target}] SUCCESS at step={step}, "
                  f"cube=({cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f})")
            if args.log_actions:
                # 只保留成功回合：以成功序号重新编号后并入日志
                action_log.extend([[n_success, s, *a] for s, *a in episode_log])
            episode_log = []

        # 超时：本回合计为失败
        elif step >= MAX_STEPS:
            reason = classify_failure(env, checker, cube_init_z, ever_lifted)
            fail_categories[reason] = fail_categories.get(reason, 0) + 1
            cube = env.env.get_p_body(CUBE_BODY)
            print(f"[eval {n_attempt}/{n_target}] FAIL [{reason}] step={step}, "
                  f"ever_lifted={ever_lifted}, cube=({cube[0]:.3f},{cube[1]:.3f},"
                  f"{cube[2]:.3f})")
            episode_log = []
        else:
            continue

        if n_attempt >= n_target:
            break

        cube_init_z = reset_episode()
        n_attempt += 1
        step = 0
        ever_lifted = False

    env.env.close_viewer()

    # 逐帧 action 关节角日志落盘（joint1~6 单位 rad，gripper 为 0/1 指令）
    if args.log_actions and action_log:
        log_path = Path(args.log_actions)
        if log_path.parent and not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "step",
                             "joint1", "joint2", "joint3",
                             "joint4", "joint5", "joint6", "gripper"])
            writer.writerows(action_log)
        print(f"action log saved to {log_path} ({len(action_log)} rows)")

    n_fail = n_attempt - n_success
    print("\n" + "=" * 60)
    print(f"EVAL RESULT: {n_success}/{n_attempt} success "
          f"= {n_success / n_attempt:.1%}")
    if n_fail:
        print("failure breakdown:")
        for k in ["GRASP_FAIL", "DROPPED", "PLACE_OFFSET",
                  "NOT_RELEASED", "TIMEOUT_OTHER"]:
            if fail_categories.get(k):
                print(f"  {k:14s} {fail_categories[k]:3d}  "
                      f"({fail_categories[k] / n_fail:.0%} of failures, "
                      f"{fail_categories[k] / n_attempt:.0%} of all)")
    print("=" * 60)


if __name__ == "__main__":
    main()
