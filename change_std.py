"""
取消文化：社會異質性（標準差）對炎上規模的影響測試
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
NUM_IGNITE = 5

B = 1.8            # 保持較高的煽動性
R = -0.1           # 保持沉默的壓力
FIXED_KAPPA = 0.6  # 固定演算法強度為 0.6 (有一定推播能力)
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

# 預先讀取真實數據的基礎平均分佈
print(f"[{time.strftime('%H:%M:%S')}] 讀取真實數據並計算基礎 Beta 參數...")
poster_ratio, beta_private_raw, beta_public_raw = load_survey_and_fit(CSV_PATH)
n_posters = round(N * poster_ratio)
n_lurkers = N - n_posters

print(f"[{time.strftime('%H:%M:%S')}] 建立小世界網路底層 (N={N})...")
G_base = nx.watts_strogatz_graph(N, k=K, p=P, seed=42)

# ==========================================
# 將單次模擬封裝成函數，方便我們傳入不同的 d 值
# ==========================================
def run_experiment_for_k(k_multiplier):
    # 1. 根據 d 值計算新的 Alpha, Beta (平均值不變，標準差改變)
    a4, b4 = beta_private_raw[0] * k_multiplier, beta_private_raw[1] * k_multiplier
    a5, b5 = beta_public_raw[0] * k_multiplier, beta_public_raw[1] * k_multiplier

    # 2. 分配角色
    roles = np.array(["Poster"] * n_posters + ["Lurker"] * n_lurkers)
    rng.shuffle(roles)

    # 3. 抽樣門檻值
    c_private = rng.beta(a4, b4, size=N)
    c_public = np.full(N, np.nan)
    poster_idx = np.where(roles == "Poster")[0]
    c_public[poster_idx] = rng.beta(a5, b5, size=n_posters)

    # 4. KOL 設定
    G = G_base.copy()
    kol_idx = rng.choice(N, size=NUM_KOLS, replace=False)
    roles[kol_idx] = "Poster"
    c_public[kol_idx] = rng.uniform(abs(R) + 0.02, 0.50, size=NUM_KOLS)

    for kol in kol_idx:
        target_nodes = rng.choice(N, size=KOL_REACH, replace=False)
        for target in target_nodes:
            if kol != target:
                G.add_edge(kol, target)

    poster_idx = np.where(roles == "Poster")[0]
    neighbors = [np.array(list(G.neighbors(i))) for i in range(N)]

    # 5. 賽局模擬核心
    state = np.zeros(N, dtype=int)
    ignite_kols = rng.choice(kol_idx, size=NUM_IGNITE, replace=False)
    state[ignite_kols] = 1 

    for _ in range(60): # 最多跑 60 步
        phi_local = np.zeros(N)
        for i in range(N):
            if len(neighbors[i]) > 0:
                phi_local[i] = state[neighbors[i]].mean()
                
        poster_mask = roles == "Poster"
        phi_global = state[poster_mask].mean() if poster_mask.sum() > 0 else 0.0

        s_new = state.copy()
        for i in range(N):
            if state[i] == 1: continue 
            
            if roles[i] == "Lurker":
                effective_threshold = (c_private[i] + R) / B
                if phi_local[i] >= effective_threshold:
                    s_new[i] = 1
            else:
                effective_threshold = (c_public[i] + R) / B
                phi_perceived = (1 - FIXED_KAPPA) * phi_local[i] + FIXED_KAPPA * phi_global
                if phi_perceived >= effective_threshold:
                    s_new[i] = 1
                    
        if np.array_equal(s_new, state): 
            break
        state = s_new
        
    return state.mean()

# ==========================================
# 執行迴圈：測試不同標準差 (k_multiplier)
# ==========================================
print(f"[{time.strftime('%H:%M:%S')}] 開始掃描社會異質性參數 (這需要大約 1 分鐘)...")

# 設定 d 的測試範圍：
# 從 0.1 (超級異質/標準差極大) 到 5.0 (超級同質/標準差極小)
k_values = np.linspace(0.1, 5.0, 20)
results = []

for k in k_values:
    final_size = run_experiment_for_k(k)
    results.append(final_size)
    print(f"  測試 k={k:.2f} (越小代表變異數越大) -> 炎上規模: {final_size:.1%}")

print(f"[{time.strftime('%H:%M:%S')}] 模擬完成！正在繪製圖表...")

# ==========================================
# 繪製結果圖
# ==========================================
plt.figure(figsize=(9, 6))

# 這裡把 X 軸畫出來，但為了讓人直觀理解，我們在圖表上註明 d 值的意義
plt.plot(k_values, results, "-o", color="darkorange", lw=3, markersize=8)

plt.title("標準差對炎上爆發的影響\n(d 越大 標準差越大)", fontsize=14)
plt.xlabel("d 乘數 (Beta分佈參數放大倍率)", fontsize=12)
plt.ylabel("最終參與炎上的比例", fontsize=12)
plt.ylim(0, 1.05)


plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()