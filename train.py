"""
Ant-v4 SAC Training Script
Trains the MuJoCo Ant environment using Soft Actor-Critic (SAC).
Monitors for stability and auto-saves when the agent stabilizes.
"""

import os
import json
import time
import argparse
import sys
import numpy as np
from datetime import datetime
from collections import deque

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")
PROGRESS_F = os.path.join(LOGS_DIR, "progress.json")
STATUS_F   = os.path.join(LOGS_DIR, "status.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ── Stability detection ────────────────────────────────────────────────────────
class StabilityDetector:
    """
    Declares stability when the mean reward over a sliding window
    exceeds `threshold` and the coefficient of variation stays low.
    """
    def __init__(self, window=10, threshold=3000.0, cv_max=0.15):
        self.window    = window
        self.threshold = threshold
        self.cv_max    = cv_max
        self.rewards   = deque(maxlen=window)

    def update(self, reward):
        self.rewards.append(reward)

    def is_stable(self):
        if len(self.rewards) < self.window:
            return False
        arr  = np.array(self.rewards)
        mean = arr.mean()
        cv   = arr.std() / (abs(mean) + 1e-8)
        return bool(mean >= self.threshold and cv <= self.cv_max)

    def stats(self):
        if not self.rewards:
            return {"mean": 0, "std": 0, "cv": 0}
        arr = np.array(self.rewards)
        mean = float(arr.mean())
        std  = float(arr.std())
        return {"mean": mean, "std": std, "cv": std / (abs(mean) + 1e-8)}


# ── Training callback ──────────────────────────────────────────────────────────
class AntTrainingCallback(BaseCallback):
    def __init__(self, eval_env, stability: StabilityDetector,
                 eval_freq=10_000, verbose=1):
        super().__init__(verbose)
        self.eval_env   = eval_env
        self.stability  = stability
        self.eval_freq  = eval_freq
        self.ep_rewards = []
        self.history    = []          # list of {"step", "reward", "stable"}
        self._stable_saved = False

    # ── helpers ────────────────────────────────────────────────────────────────
    def _evaluate(self, n_episodes=5):
        rewards = []
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            done, total = False, 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, _ = self.eval_env.step(action)
                done   = terminated or truncated
                total += r
            rewards.append(total)
        return float(np.mean(rewards))

    def _write_progress(self, mean_reward, stable):
        stats = self.stability.stats()
        entry = {
            "step":        int(self.num_timesteps),
            "reward":      round(mean_reward, 2),
            "mean_window": round(stats["mean"], 2),
            "std_window":  round(stats["std"],  2),
            "cv":          round(stats["cv"],   4),
            "stable":      stable,
            "timestamp":   datetime.utcnow().isoformat(),
        }
        self.history.append(entry)
        with open(PROGRESS_F, "w") as f:
            json.dump(self.history, f)
        return entry

    def _write_status(self, phase, entry):
        status = {
            "phase":      phase,        # "training" | "stable" | "done"
            "latest":     entry,
            "model_path": os.path.join(MODELS_DIR, "stable_ant") if phase != "training" else None,
        }
        with open(STATUS_F, "w") as f:
            json.dump(status, f)

    # ── SB3 hooks ──────────────────────────────────────────────────────────────
    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            mean_reward = self._evaluate()
            self.stability.update(mean_reward)
            stable = self.stability.is_stable()

            entry = self._write_progress(mean_reward, stable)
            phase = "training"

            if stable and not self._stable_saved:
                path = os.path.join(MODELS_DIR, "stable_ant")
                self.model.save(path)
                self._stable_saved = True
                phase = "stable"
                print(f"\n✅  STABLE at step {self.num_timesteps} "
                      f"| mean={self.stability.stats()['mean']:.1f} "
                      f"| saved → {path}.zip\n")
            elif self._stable_saved:
                phase = "stable"

            self._write_status(phase, entry)

            if self.verbose:
                s = self.stability.stats()
                print(f"[{self.num_timesteps:>8}]  "
                      f"ep_reward={mean_reward:>8.1f}  "
                      f"win_mean={s['mean']:>8.1f}  "
                      f"cv={s['cv']:.3f}  "
                      f"{'🟢 STABLE' if stable else '🔵 training'}")
        return True

    def _on_training_end(self):
        path  = os.path.join(MODELS_DIR, "final_ant")
        self.model.save(path)
        entry = self.history[-1] if self.history else {}
        self._write_status("done", entry)
        print(f"\n🏁  Training finished → {path}.zip")


# ── Main ───────────────────────────────────────────────────────────────────────
def train(total_steps=1_000_000,
          eval_freq=10_000,
          threshold=3000.0,
          learning_rate=3e-4):

    print("=" * 60)
    print("  MuJoCo Ant-v4  |  SAC Training")
    print("=" * 60)
    print(f"  total_steps  : {total_steps:,}")
    print(f"  eval_freq    : {eval_freq:,}")
    print(f"  threshold    : {threshold}")
    print(f"  learning_rate: {learning_rate}")
    print("=" * 60)

    # environments
    train_env = make_vec_env("Ant-v4", n_envs=4)
    eval_env  = gym.make("Ant-v4")

    # model
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=0,
        tensorboard_log=LOGS_DIR,
    )

    stability = StabilityDetector(window=10, threshold=threshold)
    callback  = AntTrainingCallback(eval_env, stability, eval_freq=eval_freq)

    # write initial status
    with open(STATUS_F, "w") as f:
        json.dump({"phase": "training", "latest": {}, "model_path": None}, f)

    model.learn(total_timesteps=total_steps, callback=callback)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps",     type=int,   default=1_000_000)
    p.add_argument("--eval-freq", type=int,   default=10_000)
    p.add_argument("--threshold", type=float, default=3000.0)
    p.add_argument("--lr",        type=float, default=3e-4)
    args = p.parse_args()

    train(total_steps=args.steps,
          eval_freq=args.eval_freq,
          threshold=args.threshold,
          learning_rate=args.lr)
