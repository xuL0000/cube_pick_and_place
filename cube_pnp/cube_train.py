#!/usr/bin/env python
"""在 cube 演示数据集上训练 ACT 策略（精简核心功能：支持微调续训）。

迁移自根目录 3.train.py，改动：
  1. 数据集/仓库名/输出目录指向 cube_pnp。
  2. 双视角输入（agentview + wrist）固定，全部 episode 参与训练
     （已移除验证集划分——val loss 是开环指标，不预测直播成功率）。
  3. 支持 --init_from 微调：从已有 checkpoint 加载权重，在合并数据集
     （如 DAgger 接管数据追加后）上用小学习率短步数续训，避免全量重训。

Usage:
    # ---- 从零训练（chunk_size=20，双相机：agentview + wrist）----
    python cube_train.py --steps 4000 --ckpt ./ckpt/act_cube_wrist

    # ---- 微调续训（DAgger 数据追加后推荐）----
    python cube_train.py --init_from ./ckpt/act_cube_c20 \
        --ckpt ./ckpt/act_cube_dagger2 --steps 2000 --lr 5e-5
"""
import os
import sys
import time
import argparse
from pathlib import Path

# 相对路径以 cube_pnp/ 为准，mujoco_env 等仍从项目根目录导入
os.chdir(Path(__file__).resolve().parent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torchvision import transforms
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset, LeRobotDatasetMetadata)
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps


# ===========================================================================
# 固定配置（不再暴露为命令行参数）
# ===========================================================================
REPO_NAME = "omy_cube_pnp"
NUM_WORKERS = 4
LOG_FREQ = 100
SEED = 0


# ===========================================================================
# 图像增强（与原脚本一致：高斯噪声 + clamp）
# ===========================================================================
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


# ===========================================================================
# 帧级采样器：枚举全部 episode 的帧，shuffle=True 时每轮打散
# ===========================================================================
class EpisodeFrameSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_indices, shuffle=False,
                 seed: int = 0):
        self.frame_ids = []
        for ep in episode_indices:
            lo = dataset.episode_data_index["from"][ep].item()
            hi = dataset.episode_data_index["to"][ep].item()
            self.frame_ids.extend(range(lo, hi))
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        ids = self.frame_ids
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            ids = [ids[i] for i in torch.randperm(len(ids), generator=g).tolist()]
            self.epoch += 1
        return iter(ids)

    def __len__(self):
        return len(self.frame_ids)


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Train ACT on cube demos")
    parser.add_argument("--root", type=str, default="./demo_data")
    parser.add_argument("--ckpt", type=str, default="./ckpt/act_cube")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--init_from", type=str, default=None,
                        help="已有 checkpoint 路径，加载其权重作为初始化；"
                             "省略则从零训练。归一化统计自动使用当前 --root "
                             "数据集的统计值")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # ------------------------------------------------------------------ data
    dataset_metadata = LeRobotDatasetMetadata(REPO_NAME, root=args.root)
    n_episodes = int(dataset_metadata.total_episodes)
    print(f"episodes total={n_episodes} (all used for training)")

    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: ft for k, ft in features.items()
                       if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items()
                      if k not in output_features}
    print("input: agentview + wrist dual cameras")

    cfg = ACTConfig(input_features=input_features,
                    output_features=output_features,
                    chunk_size=20, n_action_steps=10)
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

    transform = transforms.Compose([
        AddGaussianNoise(mean=0.0, std=0.02),
        transforms.Lambda(lambda x: x.clamp(0, 1)),
    ])
    dataset = LeRobotDataset(REPO_NAME,
                             delta_timestamps=delta_timestamps,
                             root=args.root,
                             image_transforms=transform)

    train_sampler = EpisodeFrameSampler(dataset, list(range(n_episodes)),
                                        shuffle=True, seed=SEED)
    train_loader = torch.utils.data.DataLoader(
        dataset, num_workers=NUM_WORKERS, batch_size=args.batch_size,
        sampler=train_sampler, pin_memory=device.type != "cpu", drop_last=True)
    print(f"train frames={len(train_sampler)}")

    # ---------------------------------------------------------------- model
    if args.init_from:
        # 微调：加载已有 checkpoint 权重；归一化统计用当前数据集
        policy = ACTPolicy.from_pretrained(
            args.init_from, config=cfg,
            dataset_stats=dataset_metadata.stats)
        print(f"finetune: init from {args.init_from} (lr={args.lr})")
    else:
        policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
        print("train from scratch")
    policy.train()
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # -------------------------------------------------------------- training
    step = 0
    done = False
    t_start = time.time()      # 训练计时：日志中输出耗时与剩余时间估计

    def fmt_duration(sec: float) -> str:
        h, rem = divmod(int(sec), 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    while not done:
        for batch in train_loader:
            inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                   for k, v in batch.items()}
            loss, loss_dict = policy.forward(inp)
            loss_parts = (f"l1 {loss_dict['l1_loss']:.4f}  "
                          f"kld*{policy.config.kl_weight:.0f} "
                          f"{loss_dict.get('kld_loss', 0.0):.4f}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % LOG_FREQ == 0:
                elapsed = time.time() - t_start
                eta = elapsed / max(step, 1) * (args.steps - step)
                print(f"step {step:5d}/{args.steps}  loss {loss.item():.4f} "
                      f"[{loss_parts}]  elapsed {fmt_duration(elapsed)}  "
                      f"ETA {fmt_duration(eta)}", flush=True)
            step += 1
            if step >= args.steps:
                done = True
                break

    print(f"FINAL train loss {loss.item():.4f}"
          f"  [training time {fmt_duration(time.time() - t_start)}]")

    # ------------------------------------------------------------- save ckpt
    ckpt_dir = Path(args.ckpt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(ckpt_dir))
    print(f"checkpoint saved to {ckpt_dir}")
    print("DONE")


if __name__ == "__main__":
    main()
