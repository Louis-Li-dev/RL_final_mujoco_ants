# 🐜 Ant RL 專案開發對話紀錄 (Development Transcript)

本文件還原了開發此「MuJoCo Ant-v4 強化學習與 3D 互動控制儀表板」專案時，使用者（User）與 AI 助理（Antigravity）之間在各個階段的合作與討論對話。

---

## 🛠️ 第一階段：專案構想與環境配置

**User:**
你好！我想要在 MuJoCo 的 `Ant-v4` 螞蟻機器人環境上，使用 Stable-Baselines3 訓練一個 Soft Actor-Critic (SAC) 模型。
我希望有兩個主要功能：
1. 訓練腳本：除了正常訓練，它還需要能夠自動偵測模型是否已經「穩定（Stable）」。一旦穩定就自動儲存權重，這樣就不用硬性等 100 萬步訓練完。
2. 網頁儀表板：用 Flask 寫一個漂亮的網頁介面。這個介面要能即時呈現 3D 模擬畫面、各種物理數據（像是座標、單步與累積回報、關節扭力），還能隨時切換加載不同的模型。
我們能開始嗎？

**Antigravity:**
沒問題！這是一個非常實用且富含挑戰性的強化學習（RL）端到端專案。
我們首先需要定義好開發的流程：
1. **環境依賴配置**：建立 `requirements.txt`。因為這是 Windows 環境上的 MuJoCo 模擬，我們需要特別說明 PyTorch (CUDA/CPU) 和 `mujoco` / `gymnasium[mujoco]` 的安裝順序。
2. **核心訓練模組 (`train.py`)**：撰寫 SAC 訓練主程式，並自訂一個 `AntTrainingCallback`，實作滑動窗口（Sliding Window）的平均回報與變異係數（CV）檢測，來判定訓練是否已經穩定。
3. **網頁控制中樞 (`app.py`)**：用 Flask 設計 API 及整合式的儀表板。為了在網頁上看到 3D 動態，我們可以用 OpenCV 在後台將每個實體步（Step）渲染出的 `rgb_array` 儲存為 JPG 圖片，並在網頁前端實作一個「影格播放器」，支援前進、後退、調速與拖曳進度條。
4. **前端視覺化**：設計精美的繁體中文儀表板，具備 Glassmorphism 現代科技感，並使用 HTML5 Canvas 或 CSS Bars 來顯示螞蟻 8 個關節的即時扭矩（Torque）。

我們先來建立 `requirements.txt` 吧！

**User:**
好，請幫我寫 `requirements.txt`，並把 PyTorch 的安裝指令也作為註解放進去。

*(Antigravity 建立了 requirements.txt)*

---

## 🏃 第二階段：編寫自動穩定偵測訓練腳本 (`train.py`)

**User:**
接著我們來寫 `train.py`。
關於「穩定偵測」，細節應該怎麼做？
一般來說，Ant-v4 如果走得好，平均回報（Reward）大約在 3000 到 6000 之間。我希望判定標準是：
- 檢查最近 10 次評估（滑動窗口大小 = 10）。
- 平均回報必須大於或等於 `3000.0`。
- 為了確保沒有大幅度晃動，這 10 次的變異係數（Coefficient of Variation, CV = 標準差 / 平均值）必須小於 `0.15`。
如果符合，就保存為 `stable_ant.zip`，並標記 `stable = True`。最後訓練結束時，不管有沒有穩定都再存一個 `final_ant.zip`。

**Antigravity:**
這個穩定偵測設計非常科學！變異係數（CV）能很好地過濾掉那些「偶爾拿到高分但波動極大」的隨機策略。
我將為您設計：
1. `StabilityDetector` 類別：維護一個 `deque` 來記錄滑動窗口內的回報，並計算 `mean`、`std` 與 `cv`。
2. `AntTrainingCallback` 類別：繼承自 Stable-Baselines3 的 `BaseCallback`。在每個評估週期（例如每 10,000 步）進行一次評估，更新 `StabilityDetector`，並將進度寫入 `logs/progress.json`，狀態寫入 `logs/status.json`。
3. `train()` 主函數：初始化 VecEnv，設定 SAC 的 hyperparameters（包含 `buffer_size=1,000,000`、`batch_size=256` 等），然後啟動 `model.learn`。

這就為您寫入 `train.py`。

*(Antigravity 建立了 train.py)*

**User:**
看起來很棒！這樣我們就可以用 `python train.py --steps 1000000` 啟動訓練，它會自動把 tensorboard 寫入 `logs/`，並即時更新 `progress.json`。

---

## 🌐 第三階段：Flask 後端 API 設計與 3D 模擬渲染 (`app.py`)

**User:**
現在我們要開發網頁儀表板。後端採用 Flask。
我需要這幾個 API 路由：
- `GET /`：回傳主頁面。
- `GET /api/status`：讀取 `logs/status.json`，告訴前端模型是否在訓練中，以及目前加載了什麼模型。
- `GET /api/progress`：讀取 `logs/progress.json` 以獲取訓練曲線數據。
- `GET /api/models`：掃描 `models/` 目錄下的所有 `.zip` 檔案，返回列表。
- `POST /api/load_model`：根據傳入的名稱加載指定的模型（例如 `sac-ant-v4`、`stable_ant` 等）。
- `POST /api/rollout`：這是最關鍵的。當使用者點擊「執行模擬」時，後端要用加載好的模型載入 `Ant-v4` 環境（`render_mode="rgb_array"`），跑一次 Rollout Episode（預設 1000 步）。在每一步：
  1. 呼叫 `model.predict` 決定動作。
  2. 呼叫 `env.step` 執行動作，並取得 `reward`。
  3. 從 MuJoCo `qpos` 取得軀幹（Torso）的 X/Y 座標。
  4. 從 `action` 向量（長度為 8）取得 8 個關節的扭矩。
  5. 呼叫 `env.render()` 渲染畫面，並用 OpenCV 保存到 `static/rollout_frames/frame_{step}.jpg`。
  6. 返回包含整條軌跡（Trajectory）數據的 JSON，讓前端可以做逐影格播放。
另外，為了方便使用者測試，如果本機還沒有任何模型，可以支援自動從 Hugging Face（`jren123/sac-ant-v4`）下載預先訓練好的模型！

**Antigravity:**
這個架構設計得非常清晰。將 3D 物理環境與網頁端連結的關鍵，就是透過將每一影格渲染並保存為實體圖片，前端再透過非同步預載入（Pre-load）來達成流暢的 3D 播放。這樣一來，我們就完全避免了在網頁端運行 WebGL 重建 MuJoCo 複雜碰撞體的困難，且能夠百分之百還原真實的物理渲染畫面。
我會注意以下幾點：
1. **動態下載**：如果選擇 `sac-ant-v4`，自動下載 Hugging Face 的權重。
2. **記憶體管理**：在每次 Rollout 開始前，先清空舊的 `static/rollout_frames` 目錄，避免硬碟空間被歷史圖片塞滿。
3. **影像轉碼**：`env.render()` 返回的是 RGB 陣列，而 OpenCV 寫入需要轉為 BGR 格式（`cv2.cvtColor`），並將 JPEG 品質設定為 85，在畫質與傳輸速率間取得完美平衡。

我會將這些後端邏輯寫入 `app.py`。

---

## 🎨 第四階段：打造中文儀表板與關節扭力圖 (`app.py` Frontend)

**User:**
後端 API 寫好了。現在我們來處理前端 HTML/CSS/JS。
我希望儀表板長這樣：
- **整體視覺**：使用極簡暗調或高質感的現代風格（和諧的 HSL 色彩、細緻的邊框、乾淨的卡片佈局與圓角）。
- **左側欄**：3D 環境實體畫面播放器。包含【播放/暫停】、【重設】按鈕、時間進度條（Slider）、播放速度選擇器（0.5x, 1x, 2x, 5x）。下方是預載好的 3D 圖片展示區。
- **右側欄上部**：模型管理器。可以選擇本地 zip 模型或預訓練模型，並設定 deterministic 選項與模擬步數。
- **右側欄下部**：即時 Telemetry 物理儀表板。呈現當前步數、單步回報、累積回報、X 座標、Y 座標、動作強度。
- **關節扭矩反饋**：呈現 Ant-v4 螞蟻機器人 8 個關節的即時扭矩。Ant 有 4 條腿，每條腿有 2 個關節（臀部 Hip、踝部 Ankle）。希望可以用進度條（Progress Bar）的形式展示，左右方向（-1.0 到 1.0）延伸，正扭矩顯示紫色/藍色，負扭矩顯示橘色。
- **日誌板**：最下方是即時活動日誌，用以回報模型載入、Rollout 狀態等。
全部介面文字請使用繁體中文！

**Antigravity:**
這將會是一個非常令人驚豔的網頁控制面板！
我會使用 CSS 自訂變數（`:root`）設計一組和諧的冷色調（深藍、亮靛紫、薄荷綠、警示橘），搭配 Google  Fonts 的 `Outfit` 與 `Noto Sans TC` 字體。
對於關節扭矩（Joint Torques）的可視化，我會建立一個由中間向兩側伸展的雙向進度條：
- 容器寬度固定，中間劃一條 `bar-midline` 作為 0 點基準。
- 透過 JS 動態修改 `.bar-fill` 的 `left` 與 `width` 屬性。當扭矩 $T \ge 0$ 時，`left` 固定在 50%，寬度為 $50\% \times T$；當 $T < 0$ 時，`left` 設定為 $50\% - 50\% \times |T|$，寬度為 $50\% \times |T|$。
- 正負扭矩使用不同色彩區分，直觀展示機器人是如何在每一步施力前行。

我會把這些寫進 `app.py` 的 `DASHBOARD_HTML` 中。

*(Antigravity 完成了 app.py 的開發並測試啟動)*

**User:**
太棒了！啟動 `python app.py` 後，網頁能夠在 `http://localhost:5050` 完美運行。加載下載好的 `sac-ant-v4` 跑 3D 模擬，畫面播放非常流暢，而且擺動關節時，扭力條也會跟著晃動，很有實體感！

---

## 📈 第五階段：評估與回報曲線生成 (`evaluate_and_plot.py`)

**User:**
專案運作得很棒。不過，我現在想看看 `sac-ant-v4.zip` 這個模型的學習曲線和回報圖。
我們能畫出來嗎？

**Antigravity:**
因為 `sac-ant-v4.zip` 是從 Hugging Face 下載的成品權重，它本身並不包含訓練過程中的歷史事件日誌（TensorBoard `.tfevents` 文件僅保留在作者本機或其雲端硬碟中）。
但是，我們可以用兩種方式來為您呈現：
1. **本機訓練曲線**：我們可以讀取您本地 `logs/SAC_*` 下的 TensorBoard 日誌並畫出您自己訓練時的曲線（像是 `SAC_4` 跑了 52 萬步，回報正在上升）。
2. **預訓練模型評估圖**：我們可以直接寫一個評估腳本，加載 `sac-ant-v4.zip` 並在環境中跑 10 個 Episodes，紀錄每一步的累積回報、X 軸前進距離，然後畫出這 10 次模擬的綜合回報圖。這可以完美展示「這個穩定模型」目前各個維度的表現！

**User:**
那就幫我寫兩個繪圖腳本：
1. 本地 TensorBoard 曲線繪製：讀取 `logs/SAC_*`，並儲存為 `learning_curves.png`。
2. 預訓練模型評估繪製：載入 `sac-ant-v4.zip` 跑 10 次 Episode，儲存為 `pretrained_evaluation.png`。

**Antigravity:**
遵命！我將為您撰寫 `plot_learning_curves.py` 與 `evaluate_and_plot.py`。
它們會使用 `matplotlib` 生成高度美化的圖表（具備網格、平均線、平滑曲線、陰影區間等），並將圖表自動保存至專案目錄與 Flask 靜態資源目錄中，方便您隨時在網頁或報告中查看。

*(Antigravity 撰寫並成功執行了這兩個腳本，生成了對應的圖表並提供了詳細的評估報告)*

**User:**
圖表畫得非常漂亮！不僅能看到我們自己訓練時的進度（`SAC_4` 確實效果最好），還能一目了然預訓練模型 `sac-ant-v4` 跑 10 次的超強表現（平均得分達到了 5678.25，前進距離穩定在 277 左右，生存率 100%）。

**Antigravity:**
很高興這些圖表對您有所幫助！這整個專案整合了 **「自動化控制訓練」**、**「即時 3D 物理渲染播放器」**、**「動態遙測與關節施力條」** 和 **「深度評估繪圖工具」**，是一個非常完整的 Reinforcement Learning 教學與展示系統。祝您接下來在機器人步行模擬的研究中一切順利！
