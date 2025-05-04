# student_agent.py
import os
import glob
import collections
import random   # 只是保留，同訓練保持一致
import cv2
import gym
import numpy as np
import torch
import torch.nn as nn


# ────────────────────────────────
# 0. Network ─ 同 train.py 的 DuelingCNN
# ────────────────────────────────
class DuelingCNN(nn.Module):
    """(C,84,84) → Q-values"""
    def __init__(self, in_shape, n_actions):
        super().__init__()
        C, H, W = in_shape
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, C, H, W)).view(1, -1).size(1)

        self.value_stream     = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(),
                                              nn.Linear(512, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(),
                                              nn.Linear(512, n_actions))

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        v = self.value_stream(x)
        a = self.advantage_stream(x)
        return v + a - a.mean(1, keepdim=True)


# ────────────────────────────────
# 1. Agent
# ────────────────────────────────
class Agent(object):
    """
    Agent used by the online judge.
    呼叫順序：
        agent = Agent()
        obs = env.reset()
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
    """
    def __init__(self):
        # ---- 動作空間（12 個，對應 COMPLEX_MOVEMENT） ----
        self.action_space = gym.spaces.Discrete(12)

        # ---- 裝置、網路 ----
        self.device = torch.device("cpu")         # 評分只允許 CPU
        self.policy = DuelingCNN((4, 84, 84),
                                 self.action_space.n).to(self.device)
        self.policy.eval()

        # ---- 載入 checkpoint ----
        ckpt = (os.getenv("CKPT_PATH")            # 優先環境變數
                or "final.pth")                   # 或 train.py 最後輸出的檔名

        if not os.path.isfile(ckpt):              # fallback：挑 checkpoints 內最新檔
            cands = sorted(glob.glob("checkpoints/*.pth"))
            ckpt = cands[-1] if cands else None

        if ckpt and os.path.isfile(ckpt):
            self.policy.load_state_dict(torch.load(ckpt,
                                                   map_location=self.device))
            print(f"[student_agent] Loaded weights from {ckpt}")
        else:
            print("[student_agent] ⚠️  No checkpoint found – using untrained net")

        # ---- 影像 buffer：當評測端沒做 FrameStack 時自備 ----
        self._frames = collections.deque(maxlen=4)

    # ──────────────────────
    # 私有：前處理成 (4,84,84)
    # ──────────────────────
    def _preprocess(self, obs_rgb):
        """單張 RGB 幀 → 更新 frame stack，回傳 (4,84,84) uint8"""
        frame = cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2GRAY)          # → gray
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, None]                                  # (84,84,1)
        self._frames.append(frame)

        # 重置後 deque 可能未滿 4 張；用最新幀補齊
        while len(self._frames) < 4:
            self._frames.append(frame)

        stacked = np.concatenate(list(self._frames), axis=2)       # (84,84,4)
        stacked = np.moveaxis(stacked, 2, 0)                       # → (4,84,84)
        return stacked

    # ──────────────────────
    # 公開：評分系統每一步呼叫
    # ──────────────────────
    @torch.no_grad()
    def act(self, observation):
        """
        Parameters
        ----------
        observation : np.ndarray
            • 若評測端 already wrapped：shape == (4,84,84) uint8
            • 否則：原始 RGB 幀，shape ≈ (240,256,3)

        Returns
        -------
        int
            動作索引，對應 COMPLEX_MOVEMENT
        """

        # ① 判斷是否已經處理過
        if observation.ndim == 3 and observation.shape[0] == 4:
            state = observation                                # (4,84,84) uint8
        else:
            state = self._preprocess(observation)              # 自行前處理

        # ② tensor 化 (0-1)，執行前向
        state_t = torch.as_tensor(state, dtype=torch.float32,
                                  device=self.device).unsqueeze(0).div_(255)

        q_values = self.policy(state_t)
        action = q_values.argmax(1).item()                     # greedy（ε=0）
        return action
