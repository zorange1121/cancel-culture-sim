"""
取消文化：注意力衰退率 (Decay Rate) 對炎上規模的影響掃描
"""

import numpy as np
import pandas as pd
from scipy import stats
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
NUM_IGNITE = 5

B = 1.8            
R = -0.1           
FIXED_KAPPA = 0.6  
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

# 初始化網路與門檻
print(f"[{time.strftime('%H:%M:%S')}] 讀取數據與建立網路模型...")
poster_ratio, beta_private_raw, beta_public_raw = load_survey_and_fit(CSV_PATH)
n_posters = round(N * poster_ratio)
n_lurkers = N - n_posters

G_base = nx.watts_strogatz_graph(N, k=K, p=P, seed=42)
roles = np.array(["Poster"] * n_posters + ["Lurker"] * n_lurkers)
rng.shuffle(roles)

c_private_base = rng.beta(beta_private_raw[0], beta_private_raw[1], size=N)
c_public_base = np.full(N, np.nan)
poster_idx = np.where(roles == "Poster")[0]
c_public_base[poster_idx] = rng.beta(beta_public_raw[0], beta_public_raw[1], size=n_posters)

# KOL 設定
G = G_base.copy()
kol_idx = rng.choice(N, size=NUM_KOLS, replace=False)
roles[kol_idx] = "Poster"
c_public_base[kol_idx] = rng.uniform(abs(R) + 0.02, 0.50, size=NUM_KOLS)

for kol in kol_idx:
    target_nodes = rng.choice(N, size=KOL_REACH, replace=False)
    for target in target_nodes:
        if kol != target:
            G.add_edge(kol, target)

poster_idx = np.where(roles == "Poster")[0]
neighbors = [np.array(list(G.neighbors(i))) for i in range(N)]

# ==========================================
# 賽局核心：支援 Decay 率的模擬函數
# ==========================================
def run_decay_experiment(decay_rate):
    has_posted = np.zeros(N, dtype=bool)
    heat = np.zeros(N, dtype=float)
    
    # 固定火種，讓比較更公平
    rng_ignite = np.random.default_rng(99) 
    ignite_kols = rng_ignite.choice(kol_idx, size=NUM_IGNITE, replace=False)
    
    has_posted[ignite_kols] = True
    heat[ignite_kols] = 1.0

    history_cumulative = [has_posted.mean()]
    history_heat = [heat.mean()]
    
    for _ in range(60):
        phi_local = np.zeros(N)
        for i in range(N):
            if len(neighbors[i]) > 0:
                phi_local[i] = heat[neighbors[i]].mean()
                
        poster_mask = roles == "Poster"
        phi_global = heat[poster_mask].mean() if poster_mask.sum() > 0 else 0.0

        new_fired = []
        for i in range(N):
            if has_posted[i]: continue 
            
            if roles[i] == "Lurker":
                effective_threshold = (c_private_base[i] + R) / B
                if phi_local[i] >= effective_threshold:
                    new_fired.append(i)
            else:
                effective_threshold = (c_public_base[i] + R) / B
                phi_perceived = (1 - FIXED_KAPPA) * phi_local[i] + FIXED_KAPPA * phi_global
                if phi_perceived >= effective_threshold:
                    new_fired.append(i)
        
        # 熱度衰退
        heat = heat * decay_rate
        
        # 新發文者滿血復活
        if len(new_fired) > 0:
            heat[new_fired] = 1.0
            has_posted[new_fired] = True
            
        history_cumulative.append(has_posted.mean())
        history_heat.append(heat.mean())
        
        if len(new_fired) == 0 and heat.max() < 0.01:
            break
            
    return history_cumulative, history_heat

# ==========================================
# 實驗一：跑三種特定衰退率 (為了畫圖 A)
# ==========================================
print(f"[{time.strftime('%H:%M:%S')}] 執行實驗一：觀察單次熱度波峰...")
cum_10, heat_10 = run_decay_experiment(decay_rate=1.0)
cum_08, heat_08 = run_decay_experiment(decay_rate=0.8)
cum_05, heat_05 = run_decay_experiment(decay_rate=0.5)

# ==========================================
# 實驗二：掃描多種 Decay 率 (為了畫圖 B)
# ==========================================
print(f"[{time.strftime('%H:%M:%S')}] 執行實驗二：掃描各種衰退率對最終規模的影響 (約需 20 秒)...")
decay_range = np.linspace(0.1, 1.0, 15)  # 測試從 0.1 到 1.0 共 15 種衰退率
finals = []

for d in decay_range:
    cum, _ = run_decay_experiment(decay_rate=d)
    finals.append(cum[-1])  # 只取累積參與比例的最後一個值 (最終規模)
    print(f"  衰退率={d:.2f} -> 最終規模={cum[-1]:.1%}")

print(f"[{time.strftime('%H:%M:%S')}] 模擬完成！正在繪圖...")

# ==========================================
# 繪圖設計 (左右對比)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 圖 A: 線上即時熱度的時間序列 (波峰現象)
ax1.plot(heat_10, "-o", color="gray", alpha=0.5, label="無衰退 (永遠累積)")
ax1.plot(heat_08, "-o", color="darkorange", lw=2, label="慢速衰退 (保留80%)")
ax1.plot(heat_05, "-o", color="steelblue", lw=2, label="快速衰退 (保留50%)")

ax1.set_title("A. 社群「即時熱度」的時間週期", fontsize=13)
ax1.set_xlabel("Time Step (時間推移)", fontsize=11)
ax1.set_ylabel("即時熱度 (目前活躍討論的比例)", fontsize=11)
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)

# 圖 B: 不同衰退率對最終參與規模的相變圖
ax2.plot(decay_range, finals, "-o", color="crimson", lw=3)

ax2.set_title("B. 平台記憶力 (Decay Rate) vs 最終炎上規模", fontsize=13)
ax2.set_xlabel("衰退保留率 (數值越小代表遺忘越快)", fontsize=11)
ax2.set_ylabel("最終參與集體攻擊的比例", fontsize=11)
ax2.set_ylim(0, max(finals) * 1.1)

# 加上輔助解說文字
ax2.axvline(x=0.7, color='black', linestyle='--', alpha=0.5, label="臨界衰退率 (Tipping Point)")
ax2.text(0.2, ax2.get_ylim()[1]*0.1, "資訊過載 / 退燒極快\n(無法形成連鎖反應)", fontsize=10, color="gray", ha='left')
ax2.text(0.95, ax2.get_ylim()[1]*0.5, "持續保溫\n(大炎上)", fontsize=10, color="darkred", ha='right')

ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()