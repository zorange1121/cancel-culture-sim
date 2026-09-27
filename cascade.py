"""
取消文化：資訊瀑布 (Informational Cascades) 與證據反轉衝擊模型
基於 Bikhchandani, Hirshleifer, and Welch (1992) 的脆弱瀑布理論
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

B = 1.8               # 社會跟風權重 (越大越容易形成瀑布)
R = -0.1              # 沉默的壓力
FIXED_KAPPA = 0.6     # 演算法推播強度

# ✨ 資訊瀑布與反轉參數
PRIVATE_BIAS_MEAN = 0.2  # 初始私人信號平均 (偏向 0.2，代表多數人一開始有點相信謠言)
PRIVATE_BIAS_STD = 0.3   # 私人信號多樣性

SHOCK_TIME = 20          # 外部公共證據 (澄清文) 出現的時間點
SHOCK_STRENGTH = -2.5    # 澄清文的威力 (負值代表強力洗白受害者)
# ==========================================

def parse_pct(val):
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
print(f"[{time.strftime('%H:%M:%S')}] 初始化網路與門檻...")
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

# ✨ 分配每個人的「私人信號 (Private Signal)」
private_signals = rng.normal(PRIVATE_BIAS_MEAN, PRIVATE_BIAS_STD, size=N)

# KOL 設定 (KOL 擁有極強的初始攻擊偏見)
G = G_base.copy()
kol_idx = rng.choice(N, size=NUM_KOLS, replace=False)
roles[kol_idx] = "Poster"
c_public_base[kol_idx] = rng.uniform(abs(R) + 0.02, 0.50, size=NUM_KOLS)
private_signals[kol_idx] = 1.0 # 初始火種深信謠言

for kol in kol_idx:
    target_nodes = rng.choice(N, size=KOL_REACH, replace=False)
    for target in target_nodes:
        if kol != target:
            G.add_edge(kol, target)

poster_idx = np.where(roles == "Poster")[0]
neighbors = [np.array(list(G.neighbors(i))) for i in range(N)]

# ==========================================
# 賽局核心：三元狀態與外部衝擊
# ==========================================
def run_cascade_experiment(max_steps=50):
    # state: 1 (攻擊), -1 (護航), 0 (沉默)
    state = np.zeros(N, dtype=int)
    
    # KOL 開砲點火
    ignite_kols = rng.choice(kol_idx, size=10, replace=False)
    state[ignite_kols] = 1 

    history_attack = [np.mean(state == 1)]
    history_defend = [np.mean(state == -1)]
    
    for t in range(max_steps):
        # ✨ 計算時間點的外部衝擊信號 G(t)
        G_t = SHOCK_STRENGTH if t >= SHOCK_TIME else 0.0
        
        # 計算局部與全局熱度 (因為狀態有正負，Phi 也會自然呈現正負相抵)
        phi_local = np.zeros(N)
        for i in range(N):
            if len(neighbors[i]) > 0:
                phi_local[i] = state[neighbors[i]].mean()
                
        poster_mask = roles == "Poster"
        phi_global = state[poster_mask].mean() if poster_mask.sum() > 0 else 0.0

        s_new = np.zeros(N, dtype=int)
        
        # 遍歷所有人進行三元決策
        for i in range(N):
            effective_threshold = (c_private_base[i] if roles[i] == "Lurker" else c_public_base[i]) + R
            phi_perceived = phi_local[i] if roles[i] == "Lurker" else ((1 - FIXED_KAPPA) * phi_local[i] + FIXED_KAPPA * phi_global)
            
            # 總評價 E_i(t) = 私人信號 + 社會跟風 + 全局證據
            E_i = private_signals[i] + B * phi_perceived + G_t
            
            # 決策邏輯
            if E_i > effective_threshold:
                s_new[i] = 1      # 證據指向有罪，發文攻擊
            elif E_i < -effective_threshold:
                s_new[i] = -1     # 證據指向無辜，發文護航
            else:
                s_new[i] = 0      # 證據不足或混亂，刪文/沉默觀望
                
        history_attack.append(np.mean(s_new == 1))
        history_defend.append(np.mean(s_new == -1))
        
        # 若狀態穩定且沒有衝擊即將發生，則提早結束
        if np.array_equal(s_new, state) and t > SHOCK_TIME:
            break
        state = s_new
        
    return history_attack, history_defend

# ==========================================
# 執行實驗與繪圖
# ==========================================
print(f"[{time.strftime('%H:%M:%S')}] 執行反轉衝擊實驗...")
hist_attack, hist_defend = run_cascade_experiment(max_steps=50)

plt.figure(figsize=(10, 6))

plt.plot(hist_attack, "-o", color="crimson", lw=3, label="攻擊者比例 (+1, Cancel)")
plt.plot(hist_defend, "-o", color="steelblue", lw=3, label="護航/反擊者比例 (-1, Defend)")

# 畫出證據反轉的時間線
plt.axvline(x=SHOCK_TIME, color='black', linestyle='--', lw=2, label="強力澄清證據釋放 G(t)")


plt.title("資訊瀑布的脆弱性：外部證據反轉如何瓦解取消文化", fontsize=15, weight="bold")
plt.xlabel("Time Step (時間推移)", fontsize=12)
plt.ylabel("全網參與發聲的比例", fontsize=12)
plt.ylim(0, max(max(hist_attack), max(hist_defend)) * 1.2)
plt.legend(fontsize=12, loc="upper right")
plt.grid(alpha=0.4)

plt.tight_layout()
plt.show()
