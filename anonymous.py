"""
取消文化進階實驗：從實證數據動態計算「平台匿名性」對炎上規模的影響
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
FIXED_KAPPA = 0.6  # 固定演算法推播強度
# ==========================================

def parse_pct(val) -> float:
    s = str(val).strip()
    if not s or s.lower() == "nan": return np.nan
    s = s.replace("%", "").strip()
    if "~" in s:
        lo, hi = s.split("~")
        return (float(lo) + float(hi)) / 2 / 100
    return float(s) / 100

def extract_platform_multipliers(df):
    """從問卷數據中萃取出實名與匿名的意願分數，轉換為門檻乘數"""
    c_real_main = df['在實名/公開帳號發表主流觀點']
    c_real_sub = df['在實名/公開帳號發表非主流觀點']
    c_anon_main = df['在完全匿名平台發表主流觀點']
    c_anon_sub = df['在完全匿名平台發表非主流觀點']

    w_real = pd.concat([c_real_main, c_real_sub]).mean()
    w_anon = pd.concat([c_anon_main, c_anon_sub]).mean()
    w_base = (w_real + w_anon) / 2
    
    # 意願越高，跨越心理門檻越容易 (成反比)
    mult_anon = w_base / w_anon
    mult_real = w_base / w_real
    
    return mult_anon, mult_real, w_anon, w_real

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
    
    # 計算進階版乘數
    mult_anon, mult_real, w_anon, w_real = extract_platform_multipliers(df)
    
    return poster_ratio, (a4, b4), (a5, b5), mult_anon, mult_real, w_anon, w_real

# ==========================================
# 1. 網路與門檻初始化
# ==========================================
print(f"[{time.strftime('%H:%M:%S')}] 分析問卷數據...")
result = load_survey_and_fit(CSV_PATH)
poster_ratio, beta_private_raw, beta_public_raw = result[0], result[1], result[2]
ANON_MULT, REAL_MULT, score_anon, score_real = result[3], result[4], result[5], result[6]

print("-" * 40)
print("【數據洞察：台灣網民發言意願】")
print(f"  實名發表意願平均分數 : {score_real:.2f} / 10")
print(f"  匿名發表意願平均分數 : {score_anon:.2f} / 10")
print(f"  => 匿名平台門檻折現率 : {ANON_MULT:.2f} 倍 (更容易煽動)")
print(f"  => 實名平台門檻加乘率 : {REAL_MULT:.2f} 倍 (阻力更大)")
print("-" * 40)

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
# 2. 賽局核心：支援平台折現率
# ==========================================
def run_anonymity_experiment(platform_mode="baseline"):
    c_public = c_public_base.copy()
    
    if platform_mode == "anonymous":
        c_public[poster_idx] = np.clip(c_public[poster_idx] * ANON_MULT, 0, 1)
    elif platform_mode == "real_name":
        c_public[poster_idx] = np.clip(c_public[poster_idx] * REAL_MULT, 0, 1)
        
    c_private = c_private_base.copy()
    state = np.zeros(N, dtype=int)
    
    # 確保每個實驗的 KOL 火種都是同一批人
    rng_ignite = np.random.default_rng(99) 
    ignite_kols = rng_ignite.choice(kol_idx, size=NUM_IGNITE, replace=False)
    state[ignite_kols] = 1 

    history = [state.mean()]
    for _ in range(60):
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
                    
        history.append(s_new.mean())
        if np.array_equal(s_new, state): 
            break
        state = s_new
        
    return history

# ==========================================
# 3. 執行實驗與繪圖
# ==========================================
print("正在執行【實名制平台 (如 FB)】模擬...")
hist_realname = run_anonymity_experiment("real_name")


print("正在執行【完全匿名平台 (如 Dcard/PTT)】模擬...")
hist_anonymous = run_anonymity_experiment("anonymous")

plt.figure(figsize=(10, 6))

plt.plot(hist_anonymous, "-o", color="crimson", lw=3, label=f"匿名平台")
plt.plot(hist_realname, "-^", color="steelblue", lw=3, label=f"實名制平台")

plt.title("平台匿名性對炎上擴散規模的影響\n(固定演算法推播 κ=0.6)", fontsize=15, weight="bold")
plt.xlabel("Time Step", fontsize=12)
plt.ylabel("全網參與炎上比例", fontsize=12)
plt.ylim(0, 1.0) # 將上限設為 40%，讓曲線的對比更清晰



plt.legend(fontsize=12, loc="lower right")
plt.grid(alpha=0.4)

plt.tight_layout()
plt.show()