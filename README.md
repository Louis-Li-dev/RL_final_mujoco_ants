# MuJoCo Ant-v4 SAC RL Training & Interactive Dashboard

This repository contains an end-to-end framework for training and visualizing a Soft Actor-Critic (SAC) reinforcement learning agent on the MuJoCo **Ant-v4** environment using Stable-Baselines3. It features an interactive Flask web dashboard to monitor training metrics, load models (including pre-trained weights from Hugging Face), run 3D simulations, and analyze real-time physics telemetry.

---

## 🚀 Features

*   **Ant-v4 SAC Training (`train.py`)**: 
    *   Trains using Soft Actor-Critic (SAC).
    *   **Auto-Stability Detection**: Monitors a sliding window of evaluation rewards and automatically saves the model (`stable_ant.zip`) as soon as the agent achieves steady high-reward performance.
*   **Interactive Web Dashboard (`app.py`)**:
    *   **3D Playback Visualizer**: Re-render rollout trajectories in 3D and step through each frame with adjustable playback speeds.
    *   **Telemetry Dashboard**: Real-time display of physical attributes (current step, reward, cumulative reward, X/Y coordinates, and action norm).
    *   **Joint Torques Feedback**: Visual bars depicting the torque commands applied to all 8 joints across the ant's 4 legs.
    *   **Model Manager**: Seamlessly switch between different saved models or automatically download pre-trained weights from Hugging Face.
    *   **Live Logging**: Real-time log feeds capturing system events, downloads, and simulation statistics.

---

## 📁 Repository Structure

```text
├── models/                  # Saved model zip files (.zip)
├── logs/                    # Training logs, progress.json, and status.json
├── static/
│   └── rollout_frames/      # Rendered 3D frames of the latest simulation episode
├── app.py                   # Flask server and dashboard interface
├── train.py                 # Stable-Baselines3 SAC training script
├── requirements.txt         # Package dependencies
└── README.md                # Project documentation
```

---

## 🛠️ Installation & Setup

Ensure you have a Python environment set up (Python 3.8+ recommended).

### 1. Install PyTorch
PyTorch is a requirement for Stable-Baselines3. Depending on your hardware, install the CUDA version or CPU version:

*   **GPU (CUDA 12.4) Support (Recommended)**:
    ```bash
    pip install torch==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
    ```
*   **CPU-Only Support**:
    ```bash
    pip install torch==2.6.0
    ```

### 2. Install Dependencies
Install the remaining packages using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🎮 How to Use

### 1. Training the Agent
To start training the Ant agent from scratch:
```bash
python train.py --steps 1000000 --eval-freq 10000 --threshold 3000.0
```
*   `--steps`: Total training timesteps (default: `1,000,000`).
*   `--eval-freq`: Evaluation frequency in steps (default: `10,000`).
*   `--threshold`: The reward threshold to trigger stability auto-saving (default: `3,000.0`).
*   `--lr`: Learning rate (default: `3e-4`).

### 2. Starting the Dashboard
Launch the Flask dashboard to control evaluation and play simulations:
```bash
python app.py --port 5050
```
Open your web browser and navigate to **`http://localhost:5050`**.

Within the dashboard, you can:
1.  Select a model from the dropdown (a pre-trained Hugging Face model `sac-ant-v4` will automatically download if selected).
2.  Click **"Load Model"** (載入模型) to initialize the weights.
3.  Set the desired simulation steps and check/uncheck **Deterministic Policy**.
4.  Click **"Run 3D Simulation"** (執行 3D 模擬) to execute a rollout episode.
5.  Use the playback controls (**Play**, **Reset**, **Speed**, and **Scrubber**) to visualize the agent's movement and analyze joint torques and coordinates frame-by-frame.
