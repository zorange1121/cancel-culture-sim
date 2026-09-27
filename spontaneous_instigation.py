"""
取消文化：自發性煽動 (Spontaneous Instigation)
─────────────────────────────────────────────
核心問題：經典門檻模型假設人是「決定論」的——門檻被滿足才行動、零不確定性。
但人不是機器人。我們把決定論的「階躍函數」換成帶雜訊的「邏輯斯諦函數」：

    P(s_i = 1) = 1 / (1 + e^(-m · ΔU_i))     ,  ΔU_i = B·Φ_perceived − C_i − R

  m 越大 → 雜訊極低、極度理性 (≈ 階躍函數)
  m 越小 → 雜訊極高、衝動盲從亂選

發現：當「火種不足」時，決定論模型必然冷啟動失敗 (沒人帶頭 → 永遠沒人動)。
但只要引入行為雜訊，某些隨機誤差會「意外」產生自發性煽動者，
替原本無法跨越的啟動鴻溝點火，讓炎上從零意外爆發。

骨架沿用既有設定：小世界網路 + KOL + 問卷 Beta 門檻 + 跨溫層 κ。
"""

import numpy as np
import pandas as pd
from scipy import stats, sparse
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import time

matplotlib.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Microsoft JhengHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ==========================================
# 參數設定
# ==========================================
CSV_PATH = "群體輿論如何影響個人發言意願 (回覆) - 表單回覆 1.csv"

N = 10_000
K = 10
P = 0.1

NUM_KOLS = 50
KOL_REACH = 200

R = 0.0            # 沉默基準收益：這裡設 0，讓「冷啟動」最純粹 (沒有同儕壓力幫忙推)
FIXED_KAPPA = 0.6  # 演算法跨溫層推播強度

# ✨ 行為雜訊敏感度 m (Logit 斜率)：m 越大越理性、自發點火越罕見
# 注意：m 太小會讓「每個人每步都有不小機率亂發聲」，單向累積下會直接灌到 100%，
# 那是『純噪音淹沒』而非『自發煽動』。真正的自發煽動要 m 偏大，
# 只有門檻最低的少數人偶爾意外點火，再由級聯接手。
M_MID = 20.0       # 偏理性：自發火種稀少 (主力情境)
M_LOW = 12.0       # 雜訊較多：意外火種較多
# ==========================================


def parse_pct(val) -> float:
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return np.nan
    s = s.replace("%", "").strip()
    if "~" in s:
        lo, hi = s.split("~")
        return (float(lo) + float(hi)) / 2 / 100
    return float(s) / 100


def load_survey_and_fit(path: str):
    df = pd.read_csv(path)
    q1_col, q4_col, q5_col = df.columns[1], df.columns[7], df.columns[8]
    df["is_poster"] = df[q1_col].str.startswith("是", na=False)
    q4_raw = df.loc[df["is_poster"], q4_col].apply(parse_pct).dropna()
    q5_raw = df.loc[df["is_poster"], q5_col].apply(parse_pct).dropna()
    poster_ratio = df["is_poster"].sum() / len(df)
    eps = 1e-4
    a4, b4, _, _ = stats.beta.fit(q4_raw.clip(eps, 1 - eps), floc=0, fscale=1)
    a5, b5, _, _ = stats.beta.fit(q5_raw.clip(eps, 1 - eps), floc=0, fscale=1)
    return poster_ratio, (a4, b4), (a5, b5)


# ---------------------------------------------------------------------------
# 1. 網路與門檻初始化
# ---------------------------------------------------------------------------
print(f"[{time.strftime('%H:%M:%S')}] 讀取問卷、建立網路與 KOL...")
poster_ratio, beta_private, beta_public = load_survey_and_fit(CSV_PATH)
n_posters = round(N * poster_ratio)
n_lurkers = N - n_posters

roles = np.array(["Poster"] * n_posters + ["Lurker"] * n_lurkers)
rng.shuffle(roles)

c_private = rng.beta(beta_private[0], beta_private[1], size=N)
c_public = np.full(N, np.nan)
poster_idx = np.where(roles == "Poster")[0]
c_public[poster_idx] = rng.beta(beta_public[0], beta_public[1], size=n_posters)

G = nx.watts_strogatz_graph(N, k=K, p=P, seed=42)
kol_idx = rng.choice(N, size=NUM_KOLS, replace=False)
roles[kol_idx] = "Poster"
c_public[kol_idx] = rng.uniform(abs(R) + 0.02, 0.50, size=NUM_KOLS)
for kol in kol_idx:
    targets = rng.choice(N, size=KOL_REACH, replace=False)
    for t in targets:
        if kol != t:
            G.add_edge(kol, t)

poster_mask = roles == "Poster"

# ✨ 用稀疏鄰接矩陣將 phi_local 向量化 (取代逐節點 for 迴圈，快上百倍)
A = nx.to_scipy_sparse_array(G, nodelist=range(N), dtype=float, format="csr")
deg = np.asarray(A.sum(axis=1)).ravel()
deg[deg == 0] = 1.0

# 每個人的參與成本 C_i (門檻)：Lurker 看私下門檻，Poster 看公開門檻
C = np.where(roles == "Lurker", c_private, c_public)


# ---------------------------------------------------------------------------
# 2. 賽局核心：決定論 vs. 機率 (Logit) 兩種模式
# ---------------------------------------------------------------------------
def run_cascade(B, num_ignite, deterministic, m=M_MID,
                kappa=FIXED_KAPPA, max_steps=60, seed=None):
    """
    單向級聯 (已發聲者不再回頭)。
    deterministic=True  → 階躍函數：ΔU≥0 才發聲 (零雜訊)
    deterministic=False → 機率函數：P = 1/(1+e^(-m·ΔU))，可自發點火
    """
    r = np.random.default_rng(seed)
    state = np.zeros(N)

    # 點火：從 KOL 中挑 num_ignite 個火種 (num_ignite=0 即「無火種冷啟動」)
    if num_ignite > 0:
        ignite = r.choice(kol_idx, size=num_ignite, replace=False)
        state[ignite] = 1.0

    history = [state.mean()]
    for _ in range(max_steps):
        phi_local = (A @ state) / deg
        phi_global = state[poster_mask].mean() if poster_mask.any() else 0.0
        phi_perceived = np.where(poster_mask,
                                 (1 - kappa) * phi_local + kappa * phi_global,
                                 phi_local)
        dU = B * phi_perceived - C - R   # 發聲相對沉默的淨效用

        silent = state == 0
        if deterministic:
            fire = silent & (dU >= 0)
            if not fire.any():
                history.append(state.mean())
                break                      # 決定論下沒人動 → 永遠凍結，提早結束
        else:
            p = 1.0 / (1.0 + np.exp(-m * dU))
            fire = silent & (r.random(N) < p)

        state[fire] = 1.0
        history.append(state.mean())
        if state.all():
            break
    return np.array(history)


# ---------------------------------------------------------------------------
# 3. 實驗一：時間序列 — 無火種冷啟動 (num_ignite = 0)
#    決定論永遠是 0；雜訊卻能「無中生有」點燃炎上
# ---------------------------------------------------------------------------
print(f"[{time.strftime('%H:%M:%S')}] 實驗一：無火種冷啟動 (決定論 vs. 雜訊)...")
B_WEAK = 1.4  # 群體興趣設在臨界點附近 (決定論在無火種下完全動不了)

hist_det = run_cascade(B_WEAK, num_ignite=0, deterministic=True)
noisy_trials = [run_cascade(B_WEAK, num_ignite=0, deterministic=False,
                            m=M_MID, seed=s) for s in range(6)]
print(f"  決定論最終規模: {hist_det[-1]:.1%}")
print(f"  雜訊 6 次最終規模: {[f'{h[-1]:.0%}' for h in noisy_trials]}")

# ---------------------------------------------------------------------------
# 4. 實驗二：掃描「群體興趣強度 B」 — 雜訊把臨界點往左推
#    給少量但「不足」的火種，看雜訊能否在更低興趣下就救活炎上
# ---------------------------------------------------------------------------
print(f"[{time.strftime('%H:%M:%S')}] 實驗二：掃描群體興趣 B (約需 30 秒)...")
NUM_IGNITE_FEW = 0
B_range = np.linspace(0.5, 3.5, 16)
N_TRIALS = 5

finals_det, finals_mid, finals_low = [], [], []
for B in B_range:
    finals_det.append(run_cascade(B, NUM_IGNITE_FEW, deterministic=True)[-1])
    mid = [run_cascade(B, NUM_IGNITE_FEW, deterministic=False, m=M_MID, seed=s)[-1]
           for s in range(N_TRIALS)]
    low = [run_cascade(B, NUM_IGNITE_FEW, deterministic=False, m=M_LOW, seed=s)[-1]
           for s in range(N_TRIALS)]
    finals_mid.append(np.mean(mid))
    finals_low.append(np.mean(low))
    print(f"  B={B:.2f} | 決定論 {finals_det[-1]:.0%} | "
          f"中雜訊 {finals_mid[-1]:.0%} | 高雜訊 {finals_low[-1]:.0%}")

print(f"[{time.strftime('%H:%M:%S')}] 完成，繪圖中...")

# ---------------------------------------------------------------------------
# 5. 繪圖
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# 圖 A：無火種冷啟動的時間序列
ax1.plot(hist_det, "-", color="black", lw=2.5,
         label="決定論 (零雜訊) — 永遠凍結在 0")
for i, h in enumerate(noisy_trials):
    ax1.plot(h, "-", color="crimson", lw=1.6, alpha=0.6,
             label=f"雜訊模型 (m={M_MID:.0f}) — 自發點火" if i == 0 else None)
ax1.set_title(f"A. 無火種冷啟動：自發性煽動 (B={B_WEAK}, 0 個火種)", fontsize=13)
ax1.set_xlabel("Time Step (時間推移)")
ax1.set_ylabel("全網累積發聲比例")
ax1.set_ylim(-0.02, 1.0)
ax1.legend(loc="center right")
ax1.grid(alpha=0.3)

# 圖 B：群體興趣 B 的相變圖 — 雜訊使臨界點左移
ax2.plot(B_range, finals_mid, "-s", color="crimson", lw=2.5,
         label=f"中度雜訊 (m={M_MID:.0f})")
ax2.plot(B_range, finals_low, "-^", color="darkorange", lw=2.0,
         label=f"高度雜訊 (m={M_LOW:.0f})")
ax2.set_title(f"B. 群體興趣 B ",
              fontsize=13)
ax2.set_xlabel("社交收益係數 B (群體對事件的興趣強度)")
ax2.set_ylabel("最終參與發聲比例")
ax2.set_ylim(-0.02, 1.05)
ax2.legend(loc="lower right")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("spontaneous_instigation.png", dpi=130)
print(f"[{time.strftime('%H:%M:%S')}] 已存檔 spontaneous_instigation.png")
plt.show()
