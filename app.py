"""
Ant RL — Flask Dashboard + Interactive API
==========================================
Routes
  GET  /                       → dashboard UI
  GET  /api/status             → current training status + latest reward
  GET  /api/progress           → full reward history (JSON array)
  POST /api/rollout            → run one episode with the saved model, return trajectory
  GET  /api/models             → list saved model files
  POST /api/load_model         → load a specific model by name
"""

import os
import json
import glob
import sys
import numpy as np
import gymnasium as gym
from flask import Flask, jsonify, request, send_from_directory, render_template_string

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")
PROGRESS_F = os.path.join(LOGS_DIR, "progress.json")
STATUS_F   = os.path.join(LOGS_DIR, "status.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")
FRAMES_DIR = os.path.join(STATIC_DIR, "rollout_frames")

app = Flask(__name__)

# Lazy-loaded model
_model      = None
_model_name = None


def _download_hf_model():
    url = "https://huggingface.co/jren123/sac-ant-v4/resolve/main/SAC-Ant-v4.zip"
    dest = os.path.join(MODELS_DIR, "sac-ant-v4.zip")
    if not os.path.exists(dest):
        os.makedirs(MODELS_DIR, exist_ok=True)
        print(f"Downloading model from Hugging Face: {url} ...")
        import urllib.request
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"Downloaded model to {dest}")
        except Exception as e:
            print(f"Failed to download model: {e}")
            return False
    return True


def _clear_frames_dir():
    import shutil
    if os.path.exists(FRAMES_DIR):
        try:
            shutil.rmtree(FRAMES_DIR)
        except Exception:
            pass
    os.makedirs(FRAMES_DIR, exist_ok=True)


def _load_model(name="sac-ant-v4"):
    global _model, _model_name
    from stable_baselines3 import SAC
    
    if name == "sac-ant-v4":
        _download_hf_model()
        
    path = os.path.join(MODELS_DIR, name)
    if not path.endswith(".zip"):
        path += ".zip"
    if not os.path.exists(path):
        return False, f"Model not found: {path}"
    _model      = SAC.load(path)
    _model_name = name
    return True, path


def _ensure_model():
    """Auto-load sac-ant-v4 if nothing is loaded yet."""
    global _model
    if _model is None:
        _download_hf_model()
        ok, _ = _load_model("sac-ant-v4")
        if not ok:
            ok, _ = _load_model("stable_ant")
            if not ok:
                ok, _ = _load_model("final_ant")
    return _model is not None


# ── Helpers ────────────────────────────────────────────────────────────────────
def _read_json(path, default):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default


# ── API routes ─────────────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    status = _read_json(STATUS_F, {"phase": "idle", "latest": {}, "model_path": None})
    status["model_loaded"] = _model is not None
    status["model_name"]   = _model_name
    return jsonify(status)


@app.route("/api/progress")
def api_progress():
    history = _read_json(PROGRESS_F, [])
    return jsonify(history)


@app.route("/api/models")
def api_models():
    files = glob.glob(os.path.join(MODELS_DIR, "*.zip"))
    names = [os.path.basename(f).replace(".zip", "") for f in sorted(files)]
    return jsonify({"models": names, "active": _model_name})


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "stable_ant")
    ok, msg = _load_model(name)
    return jsonify({"ok": ok, "message": msg, "model": _model_name})


@app.route("/api/rollout", methods=["POST"])
def api_rollout():
    """Run one deterministic episode. Returns step-by-step data + renders 3D frames."""
    if not _ensure_model():
        return jsonify({"error": "No model loaded. Train first or load a model."}), 400

    data        = request.get_json(silent=True) or {}
    max_steps   = int(data.get("max_steps", 1000))
    deterministic = bool(data.get("deterministic", True))

    _clear_frames_dir()

    import cv2
    env = gym.make("Ant-v4", render_mode="rgb_array")
    obs, _ = env.reset()

    steps      = []
    total_rew  = 0.0
    step_count = 0

    for _ in range(max_steps):
        action, _ = _model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_rew  += float(reward)
        step_count += 1
        
        # Render and save 3D frame
        try:
            frame = env.render()
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(FRAMES_DIR, f"frame_{step_count - 1}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception as e:
            print(f"Error rendering/saving frame at step {step_count}: {e}")

        steps.append({
            "step":          step_count,
            "reward":        round(float(reward), 4),
            "cum_reward":    round(total_rew, 4),
            # torso x/y position is obtained from MuJoCo qpos in Ant-v4
            "x_pos":         round(float(env.unwrapped.data.qpos[0]), 4),
            "y_pos":         round(float(env.unwrapped.data.qpos[1]), 4),
            "action_norm":   round(float(np.linalg.norm(action)), 4),
            "action":        [round(float(a), 4) for a in action],
        })
        if terminated or truncated:
            break

    env.close()
    return jsonify({
        "total_reward": round(total_rew, 2),
        "steps":        step_count,
        "trajectory":   steps,
        "model":        _model_name,
        "deterministic": deterministic,
        "total_frames":  step_count
    })


# ── Dashboard ──────────────────────────────────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🐜 螞蟻強化學習 — 3D 模擬與控制儀表板</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700;900&family=Outfit:wght@300;400;500;600;700;800&display=swap');
  
  :root {
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --accent: #6366f1;
    --accent-hover: #4f46e5;
    --accent-light: #e0e7ff;
    --green: #10b981;
    --green-bg: #d1fae5;
    --yellow: #f59e0b;
    --yellow-bg: #fef3c7;
    --red: #ef4444;
    --text: #0f172a;
    --muted: #64748b;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.04), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
  }
  
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Outfit', 'Noto Sans TC', system-ui, sans-serif; min-height: 100vh; }
  
  header { background: var(--card); border-bottom: 1px solid var(--border); padding: 1.25rem 2rem; display: flex; align-items: center; gap: 1rem; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.02); }
  header h1 { font-size: 1.4rem; font-weight: 800; color: var(--text); letter-spacing: -0.02em; }
  
  .badge { padding: .3rem .8rem; border-radius: 999px; font-size: .75rem; font-weight: 700; text-transform: uppercase; }
  .badge-training { background: #dbeafe; color: #1e40af; }
  .badge-stable   { background: var(--green-bg); color: #065f46; }
  .badge-done     { background: #f3e8ff; color: #6b21a8; }
  .badge-idle     { background: #f1f5f9; color: #475569; }
  
  /* Main Content Layout */
  main { display: grid; grid-template-columns: 1.2fr 1fr; gap: 1.5rem; padding: 1.5rem 2rem; }
  .left-col { display: flex; flex-direction: column; min-height: 580px; }
  .right-col { display: flex; flex-direction: column; gap: 1.5rem; min-height: 580px; }
  
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; box-shadow: var(--shadow); transition: box-shadow 0.3s ease; }
  .card:hover { box-shadow: var(--shadow-lg); }
  .card h2 { font-size: .85rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 1rem; font-weight: 700; }
  
  button { background: var(--accent); color: #fff; border: none; border-radius: 10px; padding: .65rem 1.4rem; font-size: .9rem; font-weight: 600; cursor: pointer; transition: all .2s; box-shadow: 0 2px 4px rgba(99, 102, 241, 0.1); }
  button:hover { background: var(--accent-hover); box-shadow: 0 4px 6px rgba(99, 102, 241, 0.2); }
  button:disabled { opacity: .4; cursor: not-allowed; box-shadow: none; }
  .btn-outline { background: transparent; border: 1px solid var(--border); color: var(--text); box-shadow: none; }
  .btn-outline:hover { background: var(--bg); border-color: var(--muted); color: var(--text); box-shadow: none; }
  
  .controls { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; }
  select { background: var(--card); color: var(--text); border: 1px solid var(--border); border-radius: 10px; padding: .55rem 1rem; font-size: .9rem; font-weight: 500; outline: none; cursor: pointer; transition: border-color .2s; }
  select:focus { border-color: var(--accent); }
  
  #log { font-size: .8rem; font-family: monospace; background: var(--bg); border-radius: 10px; padding: 1.25rem; max-height: 200px; overflow-y: auto; color: #475569; border: 1px solid var(--border); }
  .log-entry { padding: .25rem 0; border-bottom: 1px dashed var(--border); }
  .log-entry:last-child { border-bottom: none; }
  
  /* Playback & Visualization */
  .playback-controls { display: flex; align-items: center; gap: 1rem; background: var(--bg); border: 1px solid var(--border); border-radius: 12px; padding: .75rem 1.25rem; margin-bottom: 1rem; }
  .img-container { display: flex; justify-content: center; align-items: center; background: #fafaf9; border: 1px solid var(--border); border-radius: 12px; flex-grow: 1; padding: 1rem; min-height: 480px; position: relative; overflow: hidden; box-shadow: inset 0 2px 4px 0 rgba(0,0,0,0.03); }
  #ant-view { height: 460px; width: 460px; border-radius: 8px; object-fit: contain; display: none; border: 1px solid #e7e5e4; }
  .img-placeholder { display: flex; flex-direction: column; justify-content: center; align-items: center; gap: .75rem; color: var(--muted); font-size: .95rem; text-align: center; }
  
  /* Telemetry Grid */
  .telemetry-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; }
  .tel-item { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: .75rem 1rem; }
  .tel-label { font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; margin-bottom: .2rem; font-weight: 700; }
  .tel-val { font-size: 1.35rem; font-weight: 800; color: var(--text); letter-spacing: -0.02em; }
  
  /* Joint Torques */
  .torque-bars-container { display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; }
  .torque-leg-group { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: .75rem 1rem; }
  .torque-leg-title { font-size: .8rem; font-weight: 800; color: var(--text); margin-bottom: .6rem; border-bottom: 1px dashed var(--border); padding-bottom: .25rem; }
  .torque-pair { display: flex; align-items: center; gap: .75rem; margin-bottom: .5rem; }
  .torque-pair:last-child { margin-bottom: 0; }
  .torque-bar-label { font-size: .72rem; color: var(--muted); min-width: 45px; font-weight: 600; }
  
  .bar-container { position: relative; height: 10px; background: #e2e8f0; border-radius: 5px; flex-grow: 1; overflow: hidden; }
  .bar-midline { position: absolute; left: 50%; top: 0; width: 1px; height: 100%; background: #94a3b8; z-index: 1; }
  .bar-fill { position: absolute; left: 50%; top: 0; height: 100%; width: 0; z-index: 2; border-radius: 3px; transition: width 0.1s ease, left 0.1s ease; }
  .torque-bar-val { font-size: .72rem; font-family: monospace; color: var(--text); min-width: 32px; text-align: right; font-weight: 600; }
  
  input[type="range"] { -webkit-appearance: none; appearance: none; height: 6px; background: #cbd5e1; border-radius: 3px; outline: none; cursor: pointer; }
  input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 14px; height: 14px; border-radius: 50%; background: var(--accent); transition: background-color .1s; }
  input[type="range"]::-webkit-slider-thumb:hover { background: var(--accent-hover); }
</style>
</head>
<body>

<header>
  <div style="font-size:1.8rem">🐜</div>
  <h1>螞蟻強化學習 — 3D 模擬與控制儀表板</h1>
  <span id="phase-badge" class="badge badge-idle">空閒</span>
  <span style="margin-left:auto;font-size:.8rem;color:var(--muted)" id="last-update">—</span>
</header>

<main>
  <!-- Left Column: Visualizer Playback -->
  <div class="left-col">
    <div class="card" style="display:flex;flex-direction:column;gap:1rem;height:100%">
      <h2>3D 環境實體畫面視覺化</h2>
      
      <div class="playback-controls">
        <button id="play-btn" onclick="togglePlayback()" disabled>▶ 播放</button>
        <button id="reset-btn" onclick="resetPlayback()" disabled>🔁 重設</button>
        <div style="flex-grow:1;display:flex;align-items:center;gap:.5rem">
          <span style="font-size:.8rem;color:var(--muted);min-width:60px;white-space:nowrap" id="progress-text">0 / 0 步</span>
          <input type="range" id="playback-slider" min="0" max="0" value="0" oninput="seekPlayback(this.value)" style="flex-grow:1" disabled>
        </div>
        <div style="display:flex;align-items:center;gap:.5rem;white-space:nowrap">
          <span style="font-size:.8rem;color:var(--muted)">速度:</span>
          <select id="speed-select" onchange="changeSpeed(this.value)">
            <option value="100">0.5x</option>
            <option value="50" selected>1.0x</option>
            <option value="25">2.0x</option>
            <option value="10">5.0x</option>
          </select>
        </div>
      </div>

      <div class="img-container">
        <div id="placeholder-box" class="img-placeholder">
          <span style="font-size:3rem">📺</span>
          <b>尚未執行模擬</b>
          <span style="max-width:300px;font-size:.85rem">請在右側「控制台」選取模型並點擊「▶ 執行 3D 模擬」以載入與播放影格。</span>
        </div>
        <img id="ant-view" alt="MuJoCo 3D Ant View" src="" />
      </div>
    </div>
  </div>

  <!-- Right Column: Controller & Telemetry -->
  <div class="right-col">
    <!-- Controls -->
    <div class="card">
      <h2>模擬評估控制台</h2>
      <div class="controls" style="margin-bottom:1rem">
        <select id="model-select"><option>sac-ant-v4</option></select>
        <button id="load-btn" class="btn-outline" onclick="loadModel()">載入模型</button>
        <button id="rollout-btn" onclick="runRollout()" disabled>▶ 執行 3D 模擬</button>
      </div>
      <div style="display:flex;gap:1.5rem;align-items:center;margin-bottom:1rem">
        <label style="font-size:.85rem;color:var(--muted);display:flex;align-items:center;gap:.3rem">
          <input type="checkbox" id="det-check" checked> 確定性策略 (Deterministic)
        </label>
        <label style="font-size:.85rem;color:var(--muted);display:flex;align-items:center;gap:.3rem">
          模擬步數 <input type="number" id="max-steps" value="200" min="50" max="1000"
            style="width:70px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:.3rem .5rem">
        </label>
      </div>
      <div id="rollout-result" style="color:var(--muted);font-size:.85rem;border-top:1px dashed var(--border);padding-top:.75rem">尚未執行模擬。</div>
    </div>

    <!-- Telemetry -->
    <div class="card" style="flex-grow:1;display:flex;flex-direction:column;gap:1rem">
      <h2>即時物理感測 Telemetry</h2>
      
      <div class="telemetry-grid">
        <div class="tel-item">
          <div class="tel-label">當前步數</div>
          <div class="tel-val" id="tel-step">—</div>
        </div>
        <div class="tel-item">
          <div class="tel-label">單步獎勵</div>
          <div class="tel-val" id="tel-reward">—</div>
        </div>
        <div class="tel-item">
          <div class="tel-label">累積獎勵</div>
          <div class="tel-val" id="tel-cum-reward">—</div>
        </div>
        <div class="tel-item">
          <div class="tel-label">X 座標位置</div>
          <div class="tel-val" id="tel-x">—</div>
        </div>
        <div class="tel-item">
          <div class="tel-label">Y 座標位置</div>
          <div class="tel-val" id="tel-y">—</div>
        </div>
        <div class="tel-item">
          <div class="tel-label">動作強度 ‖action‖</div>
          <div class="tel-val" id="tel-act-norm">—</div>
        </div>
      </div>

      <div style="border-top:1px solid var(--border);padding-top:1rem;margin-top:.5rem">
        <h3 style="font-size:.85rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:.75rem;font-weight:700">
          關節扭矩反饋 (Joint Torques)
        </h3>
        <div class="torque-bars-container">
          <!-- Leg 1 -->
          <div class="torque-leg-group">
            <div class="torque-leg-title">前左腿 (Leg 1)</div>
            <div class="torque-pair">
              <div class="torque-bar-label">臀部 (Hip)</div>
              <div class="bar-container">
                <div class="bar-midline"></div>
                <div class="bar-fill" id="bar-joint-0"></div>
              </div>
              <span class="torque-bar-val" id="val-joint-0">0.00</span>
            </div>
            <div class="torque-pair">
              <div class="torque-bar-label">踝部 (Ankle)</div>
              <div class="bar-container">
                <div class="bar-midline"></div>
                <div class="bar-fill" id="bar-joint-1"></div>
              </div>
              <span class="torque-bar-val" id="val-joint-1">0.00</span>
            </div>
          </div>
          <!-- Leg 2 -->
          <div class="torque-leg-group">
            <div class="torque-leg-title">前右腿 (Leg 2)</div>
            <div class="torque-pair">
              <div class="torque-bar-label">臀部 (Hip)</div>
              <div class="bar-container">
                <div class="bar-midline"></div>
                <div class="bar-fill" id="bar-joint-2"></div>
              </div>
              <span class="torque-bar-val" id="val-joint-2">0.00</span>
            </div>
            <div class="torque-pair">
              <div class="torque-bar-label">踝部 (Ankle)</div>
              <div class="bar-container">
                <div class="bar-midline"></div>
                <div class="bar-fill" id="bar-joint-3"></div>
              </div>
              <span class="torque-bar-val" id="val-joint-3">0.00</span>
            </div>
          </div>
          <!-- Leg 3 -->
          <div class="torque-leg-group">
            <div class="torque-leg-title">後左腿 (Leg 3)</div>
            <div class="torque-pair">
              <div class="torque-bar-label">臀部 (Hip)</div>
              <div class="bar-container">
                <div class="bar-midline"></div>
                <div class="bar-fill" id="bar-joint-4"></div>
              </div>
              <span class="torque-bar-val" id="val-joint-4">0.00</span>
            </div>
            <div class="torque-pair">
              <div class="torque-bar-label">踝部 (Ankle)</div>
              <div class="bar-container">
                <div class="bar-midline"></div>
                <div class="bar-fill" id="bar-joint-5"></div>
              </div>
              <span class="torque-bar-val" id="val-joint-5">0.00</span>
            </div>
          </div>
          <!-- Leg 4 -->
          <div class="torque-leg-group">
            <div class="torque-leg-title">後右腿 (Leg 4)</div>
            <div class="torque-pair">
              <div class="torque-bar-label">臀部 (Hip)</div>
              <div class="bar-container">
                <div class="bar-midline"></div>
                <div class="bar-fill" id="bar-joint-6"></div>
              </div>
              <span class="torque-bar-val" id="val-joint-6">0.00</span>
            </div>
            <div class="torque-pair">
              <div class="torque-bar-label">踝部 (Ankle)</div>
              <div class="bar-container">
                <div class="bar-midline"></div>
                <div class="bar-fill" id="bar-joint-7"></div>
              </div>
              <span class="torque-bar-val" id="val-joint-7">0.00</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</main>

<div style="padding:0 2rem 2rem 2rem">
  <div class="card full">
    <h2>運作日誌 (Activity Log)</h2>
    <div id="log"></div>
  </div>
</div>

<script>
// ── Logging ───────────────────────────────────────────────────────────────────
function log(msg, color='#475569') {
  const el = document.getElementById('log');
  const d  = document.createElement('div');
  d.className = 'log-entry';
  d.innerHTML = `<span style="color:#94a3b8">${new Date().toLocaleTimeString()} </span><span style="color:${color}">${msg}</span>`;
  el.prepend(d);
  if (el.children.length > 100) el.lastChild.remove();
}

// ── Status polling ─────────────────────────────────────────────────────────────
let prevPhase = null;
async function pollStatus() {
  try {
    const s = await fetch('/api/status').then(r=>r.json());

    // badge translation
    const badge = document.getElementById('phase-badge');
    let phaseChinese = s.phase;
    if (s.phase === 'idle') phaseChinese = '空閒';
    else if (s.phase === 'training') phaseChinese = '訓練中';
    else if (s.phase === 'stable') phaseChinese = '穩定且已儲存';
    else if (s.phase === 'done') phaseChinese = '訓練完成';
    
    badge.textContent = phaseChinese;
    badge.className   = `badge badge-${s.phase}`;

    if (s.phase !== prevPhase) {
      if (s.phase === 'stable') { badge.classList.add('stable-flash'); log('🟢 模型已經穩定，權重自動儲存！', '#10b981'); }
      if (s.phase === 'done')   { log('🏁 強化學習模型訓練完成。', '#8b5cf6'); }
      prevPhase = s.phase;
    }

    // model list
    const sel  = document.getElementById('model-select');
    const mods = await fetch('/api/models').then(r=>r.json());
    const cur  = sel.value;
    sel.innerHTML = mods.models.length
      ? mods.models.map(m=>`<option${m===cur?' selected':''}>${m}</option>`).join('')
      : '<option>— 尚無本機模型 —</option>';

    // enable rollout button if a model is loaded
    document.getElementById('rollout-btn').disabled = !s.model_loaded;

  } catch(e) { /* server not ready */ }
}

// ── Load model ────────────────────────────────────────────────────────────────
async function loadModel() {
  const name = document.getElementById('model-select').value;
  log(`載入模型中: ${name} …`);
  const r = await fetch('/api/load_model', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name})
  }).then(r=>r.json());
  if (r.ok) {
    log(`✅ 模型載入成功: ${r.model}`, '#10b981');
    document.getElementById('rollout-btn').disabled = false;
  } else {
    log(`❌ 載入失敗: ${r.message}`, '#ef4444');
  }
}

// ── Run rollout & render ──────────────────────────────────────────────────────
let currentTrajectory = null;
let currentSimIndex = 0;
let playbackInterval = null;
let loadedImages = [];

async function runRollout() {
  const btn = document.getElementById('rollout-btn');
  btn.disabled = true;
  btn.textContent = '⏳ 3D 渲染中…';
  log('正在執行 3D Rollout 模擬並渲染畫面，這需要數秒鐘，請稍候…', 'var(--accent)');

  const payload = {
    deterministic: document.getElementById('det-check').checked,
    max_steps:     parseInt(document.getElementById('max-steps').value),
  };

  try {
    const r = await fetch('/api/rollout', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    }).then(r=>r.json());

    if (r.error) { log('❌ 模擬出錯: ' + r.error, '#ef4444'); return; }

    log(`✅ 模擬與 3D 渲染完成！共 ${r.steps} 步，累積回報: ${r.total_reward}`, '#10b981');
    
    document.getElementById('rollout-result').innerHTML =
      `<b style="color:var(--accent)">累積評估回報: ${r.total_reward}</b> &nbsp;|&nbsp; ` +
      `模擬步數: ${r.steps} &nbsp;|&nbsp; 載入模型: ${r.model} &nbsp;|&nbsp; ` +
      `${r.deterministic?'確定性策略':'隨機性策略'}`;

    // Save trajectory globally
    currentTrajectory = r.trajectory;
    currentSimIndex = 0;
    
    // Pre-load frames
    log('正在預載入 3D 模擬影格…');
    loadedImages = [];
    let loadCount = 0;
    const tBuster = new Date().getTime();
    
    for (let i = 0; i < r.steps; i++) {
      const img = new Image();
      img.src = `/static/rollout_frames/frame_${i}.jpg?t=${tBuster}`;
      img.onload = () => {
        loadCount++;
        if (loadCount === r.steps) {
          log('✅ 3D 影格預載入完成，隨時可以播放！', '#10b981');
        }
      };
      loadedImages.push(img);
    }
    
    // Toggle element visibility
    document.getElementById('ant-view').style.display = 'block';
    document.getElementById('placeholder-box').style.display = 'none';

    // Reset playback status
    if (playbackInterval) {
      clearInterval(playbackInterval);
      playbackInterval = null;
    }
    document.getElementById('play-btn').textContent = '▶ 播放';
    
    // Enable playback controls
    document.getElementById('play-btn').disabled = false;
    document.getElementById('reset-btn').disabled = false;
    
    const slider = document.getElementById('playback-slider');
    slider.disabled = false;
    slider.max = currentTrajectory.length - 1;
    slider.value = 0;
    
    updateSimUI();
  } catch(e) {
    log('❌ 模擬失敗: ' + e.message, '#ef4444');
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ 執行 3D 模擬';
  }
}

// ── Playback Logic ────────────────────────────────────────────────────────────
function togglePlayback() {
  const btn = document.getElementById('play-btn');
  if (playbackInterval) {
    // Pause
    clearInterval(playbackInterval);
    playbackInterval = null;
    btn.textContent = '▶ 播放';
    log('暫停播放。');
  } else {
    // Play
    if (!currentTrajectory || currentTrajectory.length === 0) return;
    if (currentSimIndex >= currentTrajectory.length - 1) {
      currentSimIndex = 0;
    }
    btn.textContent = '⏸ 暫停';
    log('開始播放 3D 實體畫面。');
    
    const speed = parseInt(document.getElementById('speed-select').value) || 50;
    startPlaybackLoop(speed);
  }
}

function startPlaybackLoop(speed) {
  if (playbackInterval) clearInterval(playbackInterval);
  playbackInterval = setInterval(() => {
    if (currentSimIndex < currentTrajectory.length - 1) {
      currentSimIndex++;
      updateSimUI();
    } else {
      clearInterval(playbackInterval);
      playbackInterval = null;
      document.getElementById('play-btn').textContent = '▶ 播放';
      log('模擬播放結束。');
    }
  }, speed);
}

function changeSpeed(speedVal) {
  const speed = parseInt(speedVal);
  if (playbackInterval) {
    startPlaybackLoop(speed);
  }
}

function resetPlayback() {
  if (playbackInterval) {
    clearInterval(playbackInterval);
    playbackInterval = null;
    document.getElementById('play-btn').textContent = '▶ 播放';
  }
  currentSimIndex = 0;
  updateSimUI();
  log('重設模擬播放進度。');
}

function seekPlayback(val) {
  currentSimIndex = parseInt(val);
  updateSimUI();
}

function updateTorqueBar(index, val) {
  const bar = document.getElementById(`bar-joint-${index}`);
  const valEl = document.getElementById(`val-joint-${index}`);
  if (!bar || !valEl) return;
  
  valEl.textContent = val.toFixed(2);
  
  const pct = Math.max(-1.0, Math.min(1.0, val)) * 50;
  if (pct >= 0) {
    bar.style.left = '50%';
    bar.style.width = `${pct}%`;
    bar.style.background = 'var(--accent)';
  } else {
    bar.style.left = `${50 + pct}%`;
    bar.style.width = `${Math.abs(pct)}%`;
    bar.style.background = 'var(--yellow)';
  }
}

function updateSimUI() {
  if (!currentTrajectory || currentTrajectory.length === 0) {
    document.getElementById('ant-view').style.display = 'none';
    document.getElementById('placeholder-box').style.display = 'flex';
    return;
  }
  
  const current = currentTrajectory[currentSimIndex];
  
  // Swap images
  if (loadedImages[currentSimIndex]) {
    document.getElementById('ant-view').src = loadedImages[currentSimIndex].src;
  }
  
  // Slider and progress text
  const slider = document.getElementById('playback-slider');
  slider.value = currentSimIndex;
  document.getElementById('progress-text').textContent = `${currentSimIndex + 1} / ${currentTrajectory.length} 步`;
  
  // Telemetry updates
  document.getElementById('tel-step').textContent = current.step;
  document.getElementById('tel-reward').textContent = current.reward.toFixed(4);
  document.getElementById('tel-cum-reward').textContent = current.cum_reward.toFixed(4);
  document.getElementById('tel-x').textContent = current.x_pos.toFixed(4);
  document.getElementById('tel-y').textContent = current.y_pos.toFixed(4);
  document.getElementById('tel-act-norm').textContent = current.action_norm.toFixed(4);
  
  // Torque bars updates
  const action = current.action || [0,0,0,0,0,0,0,0];
  for (let i = 0; i < 8; i++) {
    updateTorqueBar(i, action[i]);
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────
log('3D 實體控制儀表板已啟動。正在同步服務器狀態…');
pollStatus();
setInterval(pollStatus, 3000);
updateSimUI();
</script>
</body>
</html>"""


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    print(f"\n🌐  Ant RL Dashboard → http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)
