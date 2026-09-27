"""
取消文化與演算法推播模擬 — 實驗：P（重連機率 / 網路交錯度）變化 (5次隨機種子取平均版)
"""

import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import time

# 設定中文字體
matplotlib.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Microsoft JhengHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ==========================================
# 核心實驗參數設定
# ==========================================
CSV_PATH = "群體輿論如何影響個人發言意願 (回覆) - 表單回覆 1.csv" 

N = 10000         # 網路節點總數
K = 10             # 小世界網路平均基礎連接數
# P 將在下方動態掃描
KAPPA = 0.6        # 固定的演算法跨同溫層推薦強度

NUM_KOLS = 20      # KOL 總數量
KOL_REACH = 200    # 每個 KOL 額外擁有的觸及人數
NUM_IGNITE = 5     # 點火機制

B = 1.5           # 社交收益係數
R = 0             # 沈默基準收益

SEEDS = [42, 100, 2026, 777, 999]  # 定義 5 個不同的隨機種子
# ==========================================

def parse_pct(val) -> float:
    s = str(val).strip()
    if not s or s.lower() == "nan": return np.nan
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
    a4, b4, _, _ = stats.beta.fit(q4_raw.clip(eps, 1-eps), floc=0, fscale=1)
    a5, b5, _, _ = stats.beta.fit(q5_raw.clip(eps, 1-eps), floc=0, fscale=1)
    return poster_ratio, (a4, b4), (a5, b5)

print(f"[{time.strftime('%H:%M:%S')}] Step 1: 讀取真實數據並擬合門檻分佈...")
poster_ratio, beta_private, beta_public = load_survey_and_fit(CSV_PATH)

def run_simulation_with_P(current_P, current_seed, kappa=KAPPA, max_steps=60):
    """給定特定 P 值與隨機種子，建立網路並執行模擬"""
    # 針對該次模擬初始化獨立的隨機產生器
    sim_rng = np.random.default_rng(current_seed)
    
    n_posters = round(N * poster_ratio)
    n_lurkers = N - n_posters
    roles = np.array(["Poster"] * n_posters + ["Lurker"] * n_lurkers)
    sim_rng.shuffle(roles)

    c_private = sim_rng.beta(beta_private[0], beta_private[1], size=N)
    c_public = np.full(N, np.nan)
    poster_idx = np.where(roles == "Poster")[0]
    c_public[poster_idx] = sim_rng.beta(beta_public[0], beta_public[1], size=n_posters)

    # 建立目前 P 值的小世界網路 (傳入指定的 seed)
    G = nx.watts_strogatz_graph(N, k=K, p=current_P, seed=current_seed)

    # KOL 設定
    kol_idx = sim_rng.choice(N, size=NUM_KOLS, replace=False)
    roles[kol_idx] = "Poster"
    c_public[kol_idx] = sim_rng.uniform(abs(R) + 0.02, 0.50, size=NUM_KOLS)
    
    for kol in kol_idx:
        target_nodes = sim_rng.choice(N, size=KOL_REACH, replace=False)
        for target in target_nodes:
            if kol != target:
                G.add_edge(kol, target)

    poster_mask = roles == "Poster"
    neighbors = [np.array(list(G.neighbors(i))) for i in range(N)]

    # 初始化狀態與點火
    state = np.zeros(N, dtype=int)
    ignite_kols = sim_rng.choice(kol_idx, size=NUM_IGNITE, replace=False)
    state[ignite_kols] = 1

    # 模擬疊代
    for _ in range(max_steps):
        phi_local = np.zeros(N)
        for i in range(N):
            if len(neighbors[i]) > 0:
                phi_local[i] = state[neighbors[i]].mean()
                
        phi_global = state[poster_mask].mean() if poster_mask.sum() > 0 else 0.0
        s_new = state.copy()
        
        for i in range(N):
            if state[i] == 1: continue
            if roles[i] == "Lurker":
                if phi_local[i] >= (c_private[i] + R) / B:
                    s_new[i] = 1
            else:
                phi_perceived = (1 - kappa) * phi_local[i] + kappa * phi_global
                if phi_perceived >= (c_public[i] + R) / B:
                    s_new[i] = 1
                    
        if np.array_equal(s_new, state): 
            break
        state = s_new
        
    return state.mean()

print(f"[{time.strftime('%H:%M:%S')}] Step 2: 執行實驗 (掃描重連機率 P，每個 P 跑 {len(SEEDS)} 個 Seed 取平均)...")
# P 從 0.0 到 1.0 測試 15 個點
P_range = np.linspace(0.0, 1.0, 15)
finals_P_mean = []
finals_P_std = []

for p in P_range:
    run_results = []
    for seed in SEEDS:
        final_scale = run_simulation_with_P(current_P=p, current_seed=seed)
        run_results.append(final_scale)
        
    avg_scale = np.mean(run_results)
    std_scale = np.std(run_results)
    finals_P_mean.append(avg_scale)
    finals_P_std.append(std_scale)
    print(f"  測試 P = {p:.2f} -> 平均炎上規模: {avg_scale:.1%} (Std: {std_scale:.2f})")

print(f"[{time.strftime('%H:%M:%S')}] 模擬完成！正在繪製圖表...")

# 繪圖
plt.figure(figsize=(8, 5))
plt.plot(P_range, finals_P_mean, "-o", color="forestgreen", lw=2, label="5次模擬平均值")
plt.title(f"網路重連機率 (P) vs 最終炎上規模 (固定 κ={KAPPA}, K={K}, 5次 Seed 平均)")
plt.xlabel("小世界網路隨機重連機率 (P)")
plt.ylabel("最終總發聲比例 (平均)")
plt.ylim(0, 1)

# 填滿趨勢線下方區域
plt.fill_between(P_range, 0, finals_P_mean, color='forestgreen', alpha=0.1)

# 如果想在圖表上看出 5 次實驗的波動範圍，可以取消下方這行的註解：
# plt.fill_between(P_range, np.array(finals_P_mean) - np.array(finals_P_std), np.array(finals_P_mean) + np.array(finals_P_std), color='forestgreen', alpha=0.15, label='標準差範圍')

plt.grid(alpha=0.3)
plt.legend()
plt.show()