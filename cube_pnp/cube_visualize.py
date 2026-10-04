#!/usr/bin/env python
"""cube 演示数据集的可视化与统计工具。

四种模式（互斥，默认 viewer 回放）：

  1) viewer 回放（默认）：在 MuJoCo 中按记录的关节角开环重放某一集，
     viewer 角上叠加数据集中的 agentview / wrist 双相机帧：
         python cube_visualize.py --episode 0

  2) 数据集统计（--stats）：每集帧数、7 维 action 范围、gripper 闭合命令
     分布（自适应闭合触发值）、cube/plate 初始位置空间覆盖：
         python cube_visualize.py --stats

  3) 导出图像（--plot EP）：把某集的双相机代表帧 + 7 维 action 关节角曲线
     画在一张 png 里（headless，Agg 后端）：
         python cube_visualize.py --plot 0 --out /tmp/ep0.png

  4) GT 开环回放（--replay N）：重放前 N 集，用 place 三条件判定成功/失败，
     用于核查数据集质量：
         python cube_visualize.py --replay 10
"""
import os
import sys
import glob
import argparse
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def list_parquet(root: str):
    return sorted(glob.glob(os.path.join(root, 'data', 'chunk-000', '*.parquet')))


# ===========================================================================
# 5.2 数据集统计
# ===========================================================================
def print_stats(root: str):
    files = list_parquet(root)
    n_ep = len(files)
    frames, actions, obj_inits = [], [], []
    for f in files:
        df = pd.read_parquet(f)
        frames.append(len(df))
        actions.append(np.stack(df['action'].to_numpy()))
        obj_inits.append(np.stack(df['obj_init'].to_numpy())[0])
    actions = np.concatenate(actions, axis=0)          # (总帧, 7)
    obj_inits = np.stack(obj_inits)                     # (集数, 6)
    frames = np.array(frames)

    print(f"dataset={root}")
    print(f"episodes={n_ep}  total_frames={frames.sum()}  "
          f"fps=20 -> {frames.sum()/20:.1f}s")
    print(f"\n[帧数/集] min={frames.min()} max={frames.max()} "
          f"mean={frames.mean():.1f} std={frames.std():.1f}")

    names = ['joint1', 'joint2', 'joint3', 'joint4',
             'joint5', 'joint6', 'gripper']
    print("\n[action 范围] (6 关节角 rad + gripper 命令 0~1)")
    for i, n in enumerate(names):
        col = actions[:, i]
        print(f"  {n:8s} min={col.min():+.3f} max={col.max():+.3f} "
              f"mean={col.mean():+.3f} std={col.std():.3f}")

    grip = np.round(actions[:, 6], 3)
    vals, counts = np.unique(grip, return_counts=True)
    dist = ", ".join(f"{v:.2f}:{c}" for v, c in zip(vals, counts))
    print(f"\n[gripper 命令分布] 值:帧数\n  {dist}")
    closed = grip[grip > 0.01]
    if len(closed):
        print(f"  闭合帧触发命令: min={closed.min():.2f} "
              f"max={closed.max():.2f} mean={closed.mean():.3f} "
              f"（自适应闭合每集触发值；值越大说明夹得越紧才接触）")

    cube_xy = obj_inits[:, :2]
    plate_xy = obj_inits[:, 3:5]
    print("\n[初始位置空间覆盖]")
    print(f"  cube  x[{cube_xy[:,0].min():.3f},{cube_xy[:,0].max():.3f}] "
          f"y[{cube_xy[:,1].min():.3f},{cube_xy[:,1].max():.3f}]")
    print(f"  plate x[{plate_xy[:,0].min():.3f},{plate_xy[:,0].max():.3f}] "
          f"y[{plate_xy[:,1].min():.3f},{plate_xy[:,1].max():.3f}]")
    # 唯一初始位姿数量：固定 seed 全部相同，随机数据每集不同
    uniq = len(np.unique(np.round(obj_inits, 4), axis=0))
    print(f"  唯一初始位姿数={uniq}/{n_ep} "
          f"({'固定位置' if uniq == 1 else '随机/多样位置'})")
    # cube-plate 间距分布（影响抓取-搬运难度）
    gap = np.linalg.norm(cube_xy - plate_xy, axis=1)
    print(f"  cube-plate 水平间距 min={gap.min():.3f} max={gap.max():.3f} "
          f"mean={gap.mean():.3f}")


# ===========================================================================
# 5.1 导出双相机帧 + action 曲线
# ===========================================================================
def save_plot(repo_name: str, root: str, episode: int, out_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torchvision

    dataset = LeRobotDataset(repo_name, root=root)
    lo = dataset.episode_data_index["from"][episode].item()
    hi = dataset.episode_data_index["to"][episode].item()
    n = hi - lo

    # 抽 3 帧（首/中/末）的双相机图像
    pick = [lo, (lo + hi) // 2, hi - 1]
    images = []
    for fid in pick:
        item = dataset[fid]
        # lerobot v2.1: image 项是 float tensor (3,H,W) in [0,1]
        agent = item['observation.image']
        wrist = item['observation.wrist_image']
        def to_np(t):
            if isinstance(t, torch.Tensor):
                t = t.cpu().numpy()
            return np.transpose(t, (1, 2, 0))
        images.append((to_np(agent), to_np(wrist)))

    # action 曲线
    actions, states = [], []
    for fid in range(lo, hi):
        item = dataset[fid]
        actions.append(item['action'].numpy())
        states.append(item['observation.state'].numpy())
    actions = np.stack(actions)
    states = np.stack(states)

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1.6, 1.6])
    titles = ['frame start', 'frame middle', 'frame end']
    for j, ((ag, wr), ttl) in enumerate(zip(images, titles)):
        ax1 = fig.add_subplot(gs[0, j]); ax1.imshow(ag); ax1.set_title(f'agent {ttl}')
        ax1.axis('off')
        ax2 = fig.add_subplot(gs[1, j]); ax2.imshow(wr); ax2.set_title(f'wrist {ttl}')
        ax2.axis('off')

    names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']
    axa = fig.add_subplot(gs[2, :])
    for i in range(7):
        axa.plot(actions[:, i], label=names[i])
    axa.set_title(f'episode {episode} ACTION (recorded compute_q + gripper), {n} frames')
    axa.legend(ncol=7, fontsize=8); axa.grid(alpha=0.3)

    axs = fig.add_subplot(gs[3, :])
    snames = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    for i in range(6):
        axs.plot(states[:, i], label=snames[i])
    axs.set_title('observation.state (ee pose)')
    axs.legend(ncol=6, fontsize=8); axs.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    print(f"saved: {out_path}")


# ===========================================================================
# viewer 回放（双相机 overlay，迁移自 2.visualize_data.py）
# ===========================================================================
def viewer_replay(repo_name: str, root: str, xml_path: str, episode: int):
    from mujoco_env.y_env import SimpleEnv
    from cube_collect_data import wait_until_settled

    dataset = LeRobotDataset(repo_name, root=root)

    class EpisodeSampler(torch.utils.data.Sampler):
        def __init__(self, ds, ep):
            f = ds.episode_data_index["from"][ep].item()
            t = ds.episode_data_index["to"][ep].item()
            self.frame_ids = range(f, t)
        def __iter__(self):
            return iter(self.frame_ids)
        def __len__(self):
            return len(self.frame_ids)

    sampler = EpisodeSampler(dataset, episode)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=1, batch_size=1, sampler=sampler)

    env = SimpleEnv(xml_path, action_type='joint_angle')
    env.reset()
    wait_until_settled(env)
    it = iter(dataloader)
    step = 0
    while env.env.is_viewer_alive():
        env.step_env()
        if env.env.loop_every(HZ=20):
            data = next(it)
            if step == 0:
                env.set_obj_pose(data['obj_init'][0, :3].numpy(),
                                 data['obj_init'][0, 3:].numpy())
                wait_until_settled(env)
            action = data['action'].numpy()
            env.step(action[0])
            env.rgb_agent = (data['observation.image'][0].numpy() * 255
                             ).astype(np.uint8).transpose(1, 2, 0)
            env.rgb_ego = (data['observation.wrist_image'][0].numpy() * 255
                           ).astype(np.uint8).transpose(1, 2, 0)
            env.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
            env.render()
            step += 1
            if step == len(sampler):
                it = iter(dataloader)
                env.reset()
                wait_until_settled(env)
                step = 0
    env.env.close_viewer()


# ===========================================================================
# GT 开环回放 + 三条件成功判定
# ===========================================================================
def gt_replay(root: str, xml_path: str, n_episodes: int):
    from mujoco_env.y_env import SimpleEnv
    from cube_collect_data import ContactGraspChecker, wait_until_settled, CUBE_BODY, PLATE_BODY

    files = list_parquet(root)
    n_episodes = min(n_episodes, len(files))
    env = SimpleEnv(xml_path, action_type='joint_angle')
    checker = ContactGraspChecker(env.env, min_geoms=2)

    n_ok = 0
    for ep in range(n_episodes):
        df = pd.read_parquet(files[ep])
        actions = np.stack(df['action'].to_numpy()).astype(np.float32)
        obj_init = np.stack(df['obj_init'].to_numpy())[0]

        env.reset()
        env.set_obj_pose(obj_init[:3], obj_init[3:])
        wait_until_settled(env)

        i = 0
        while env.env.is_viewer_alive() and i < len(actions):
            env.step_env()
            if not env.env.loop_every(HZ=20):
                continue
            env.grab_image()
            env.step(actions[i])
            env.render()
            i += 1
        for _ in range(40):       # 放完后等待沉降
            env.step_env()
            env.env.render()

        cube = env.env.get_p_body(CUBE_BODY)
        plate = env.env.get_p_body(PLATE_BODY)
        horiz = float(np.linalg.norm(cube[:2] - plate[:2]))
        npc = len(checker.contacting_plate_geoms(env.env))
        ngc = len(checker.contacting_gripper_geoms(env.env))
        ok = horiz < checker.plate_radius and npc >= 1 and ngc == 0
        n_ok += ok
        print(f"EP{ep:<3d} {'PASS' if ok else 'FAIL'} frames={len(actions)} "
              f"horiz={horiz:.3f} plate_contact={npc} grip_contact={ngc}")
    env.env.close_viewer()
    print(f"\nGT REPLAY: {n_ok}/{n_episodes} = {n_ok/n_episodes:.1%} open-loop success")


# ===========================================================================
# 固定配置（与其余脚本一致，不再暴露为命令行参数）
REPO_NAME = "omy_cube_pnp"
XML_PATH = "./asset/example_scene_cube.xml"


def main():
    p = argparse.ArgumentParser(description="cube dataset visualization & stats")
    p.add_argument("--root", type=str, default="./demo_data")
    p.add_argument("--episode", type=int, default=0, help="viewer 回放的集号")
    p.add_argument("--stats", action="store_true", help="打印数据集统计")
    p.add_argument("--plot", type=int, default=None, metavar="EP",
                   help="导出指定集的双相机帧 + action 曲线图（headless）")
    p.add_argument("--out", type=str, default="/tmp/cube_ep.png")
    p.add_argument("--replay", type=int, default=None, metavar="N",
                   help="GT 开环回放前 N 集并统计成功率")
    args = p.parse_args()

    if args.stats:
        print_stats(args.root)
    elif args.plot is not None:
        save_plot(REPO_NAME, args.root, args.plot, args.out)
    elif args.replay is not None:
        gt_replay(args.root, XML_PATH, args.replay)
    else:
        viewer_replay(REPO_NAME, args.root, XML_PATH, args.episode)


if __name__ == "__main__":
    main()
