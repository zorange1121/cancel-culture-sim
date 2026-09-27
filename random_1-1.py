"""
取消文化：資訊瀑布與「有限理性機率決策 (Logit Dynamics)」模型
加入 P(s) = Softmax(m * U) 的隨機性機制
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

B = 1.8               
R = -0.1              
FIXED_KAPPA = 0.6     

PRIVATE_BIAS_MEAN = 0.2  
PRIVATE_BIAS_STD = 0.3   

SHOCK_TIME = 20          
SHOCK_STRENGTH = -2.5    

# ✨ 新增：有限理性參數 (Logit 敏感度 m)
# 數值越大，代表網民越「嚴格遵循效用公式」(越接近確定性)
# 數值越小，代表網民的決策越「隨機/衝動」
M_PARAM = 15.0  
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

private_signals = rng.normal(PRIVATE_BIAS_MEAN, PRIVATE_BIAS_STD, size=N)

G = G_base.copy()
kol_idx = rng.choice(N, size=NUM_KOLS, replace=False)
roles[kol_idx] = "Poster"
c_public_base[kol_idx] = rng.uniform(abs(R) + 0.02, 0.50, size=NUM_KOLS)
private_signals[kol_idx] = 1.0 

for kol in kol_idx:
    target_nodes = rng.choice(N, size=KOL_REACH, replace=False)
    for target in target_nodes:
        if kol != target:
            G.add_edge(kol, target)

poster_idx = np.where(roles == "Poster")[0]
neighbors = [np.array(list(G.neighbors(i))) for i in range(N)]

# 預先計算每個人的底線門檻 (增加效能)
effective_thresholds = np.where(roles == "Lurker", c_private_base, c_public_base) + R

# ==========================================
# 賽局核心：Softmax 機率抽樣決策
# ==========================================
def run_probabilistic_cascade(max_steps=50):
    state = np.zeros(N, dtype=int)
    
    ignite_kols = rng.choice(kol_idx, size=10, replace=False)
    state[ignite_kols] = 1 

    history_attack = [np.mean(state == 1)]
    history_defend = [np.mean(state == -1)]
    
    for t in range(max_steps):
        G_t = SHOCK_STRENGTH if t >= SHOCK_TIME else 0.0
        
        phi_local = np.zeros(N)
        for i in range(N):
            if len(neighbors[i]) > 0:
                phi_local[i] = state[neighbors[i]].mean()
                
        poster_mask = roles == "Poster"
        phi_global = state[poster_mask].mean() if poster_mask.sum() > 0 else 0.0

        # 計算感知熱度
        phi_perceived = np.where(roles == "Lurker", phi_local, 
                                 (1 - FIXED_KAPPA) * phi_local + FIXED_KAPPA * phi_global)
        
        # 綜合評價 E_i(t)
        E_t = private_signals + B * phi_perceived + G_t
        
        # ✨ 1. 計算三種選擇的相對效用 (Utility)
        U_attack = E_t - effective_thresholds
        U_defend = -E_t - effective_thresholds
        U_silent = np.zeros(N)  # 沉默的效用基準為 0
        
        # ✨ 2. 轉換為 Softmax 機率分佈 (使用 M_PARAM 控制敏感度)
        exp_attack = np.exp(M_PARAM * U_attack)
        exp_defend = np.exp(M_PARAM * U_defend)
        exp_silent = np.exp(M_PARAM * U_silent) # 這裡其實就是 1.0
        
        total_exp = exp_attack + exp_defend + exp_silent
        
        p_attack = exp_attack / total_exp
        p_defend = exp_defend / total_exp
        # p_silent 自然就是 1 - p_attack - p_defend
        
        # ✨ 3. 根據機率進行隨機抽樣 (輪盤賭法則)
        rand_vals = rng.random(N)
        s_new = np.zeros(N, dtype=int)
        
        # 若落在 attack 的機率區間 -> 攻擊
        s_new[rand_vals < p_attack] = 1
        # 若落在 defend 的機率區間 -> 護航
        s_new[(rand_vals >= p_attack) & (rand_vals < p_attack + p_defend)] = -1
        # 剩下的人維持 0 (沉默)
                
        history_attack.append(np.mean(s_new == 1))
        history_defend.append(np.mean(s_new == -1))
        
        # 因為是機率模型，狀態會不斷有微小波動，所以取消原本的「完全不變才 break」
        # 改為：若大震盪期已過，且波動極小，才視為收斂
        if t > SHOCK_TIME + 10:
            if abs(history_attack[-1] - history_attack[-2]) < 0.001 and \
               abs(history_defend[-1] - history_defend[-2]) < 0.001:
                break
        state = s_new
        
    return history_attack, history_defend

# ==========================================
# 執行實驗與繪圖
# ==========================================
print(f"[{time.strftime('%H:%M:%S')}] 執行機率化反轉衝擊實驗...")
hist_attack, hist_defend = run_probabilistic_cascade(max_steps=50)

plt.figure(figsize=(10, 6))

plt.plot(hist_attack, "-o", color="crimson", lw=3, label="攻擊者比例 (+1, Cancel)")
plt.plot(hist_defend, "-o", color="steelblue", lw=3, label="護航/反擊者比例 (-1, Defend)")

plt.axvline(x=SHOCK_TIME, color='black', linestyle='--', lw=2, label="強力澄清證據釋放 G(t)")

# 標註區域
plt.text(SHOCK_TIME / 2, max(hist_attack)*0.8, "機率資訊瀑布\n(包含衝動性噪音)", fontsize=11, color="darkred", ha='center', bbox=dict(facecolor='white', alpha=0.7))
plt.text(SHOCK_TIME + 5, max(hist_defend)*0.8, "瀑布崩塌與逆火效應", fontsize=11, color="navy", ha='left', bbox=dict(facecolor='white', alpha=0.7))

plt.title(f"有限理性決策下的取消文化與逆轉\n(機率參數 m = {M_PARAM})", fontsize=15, weight="bold")
plt.xlabel("Time Step (時間推移)", fontsize=12)
plt.ylabel("全網參與發聲的比例", fontsize=12)
plt.ylim(0, max(max(hist_attack), max(hist_defend)) * 1.2)
plt.legend(fontsize=12, loc="upper right")
plt.grid(alpha=0.4)

plt.tight_layout()
plt.show()