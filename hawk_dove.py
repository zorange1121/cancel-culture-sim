"""
取消文化延伸：鷹鴿賽局與網路極化 (Hawk-Dove / Polarization)
─────────────────────────────────────────────────────────
場景：網路上有兩個對立的同溫層 (echo chamber)
  • A 陣營 → 傾向「攻擊 / 取消對方」(極端行動 = +1)
  • B 陣營 → 傾向「護航 / 反擊酸民」(極端行動 = -1)
  • 任何人也可以選擇當鴿派 (理性沉默 / 降溫，狀態 = 0)

演算法參數 κ：把對立陣營的貼文強制推播給你 (跨溫層交鋒)。
個體採取「自己陣營極端行動」的淨效用：

    U_i = priv_i + B_in·Φ_in(同溫層支持) + B_react·κ·Φ_opp(看到敵營動員) − C_i + R

  • Φ_in  : 鄰居中「跟我同立場」者的比例 → 同溫層的相挺 (從眾)
  • Φ_opp : 全網「敵營」動員比例 → 看到對手越兇，越激起我表態 (反彈/逆火)
  • κ 越大 → 演算法把敵營塞到你面前 → 反彈項越強 → 越容易極化

決策：
  決定論 (m→∞)        : U_i ≥ 0 就表態
  機率 (Logit, m 有限) : P(表態) = 1/(1+e^(-m·U_i))

論述重點：高 κ 把兩派拉在一起時——
  • 大家若極度理性 (決定論) → 社群急速「撕裂極化」
  • 一旦加入行為雜訊 → 極化程度被「顯著削弱」(人性的不完美阻止了絕對撕裂)
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
BRIDGE_FRAC = 0.02     # 兩個同溫層之間原生的跨群連結比例 (結構性 baseline)

NUM_KOLS = 40          # 每陣營一半，當作各自的意見領袖
KOL_REACH = 200

R = 0.0
B_IN = 0.7             # 同溫層相挺權重 (從眾)
B_REACT = 1.2          # 看到敵營的反彈權重 (逆火/被激怒)
PRIV_MEAN = 0.12       # 個體對自己陣營的基本認同強度
PRIV_STD = 0.12

M_NOISE = 2.5          # 雜訊模型的 Logit 斜率 (m 小 → 雜訊大)
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


def load_public_beta(path: str):
    """只取『公開發表門檻』的 Beta 分佈，當作大家在公開場合表態的成本。"""
    df = pd.read_csv(path)
    q1_col, q5_col = df.columns[1], df.columns[8]
    df["is_poster"] = df[q1_col].str.startswith("是", na=False)
    q5_raw = df.loc[df["is_poster"], q5_col].apply(parse_pct).dropna()
    eps = 1e-4
    a5, b5, _, _ = stats.beta.fit(q5_raw.clip(eps, 1 - eps), floc=0, fscale=1)
    return a5, b5


# ---------------------------------------------------------------------------
# 1. 建立「雙同溫層」網路
# ---------------------------------------------------------------------------
print(f"[{time.strftime('%H:%M:%S')}] 建立雙同溫層網路與門檻...")
a5, b5 = load_public_beta(CSV_PATH)

half = N // 2
G_A = nx.watts_strogatz_graph(half, k=K, p=P, seed=1)
G_B = nx.watts_strogatz_graph(N - half, k=K, p=P, seed=2)
G = nx.disjoint_union(G_A, G_B)        # B 陣營節點編號自動往後平移 half

# 兩個同溫層之間加入少量原生跨群連結
n_bridge = int(BRIDGE_FRAC * N)
for _ in range(n_bridge):
    u = rng.integers(0, half)
    v = rng.integers(half, N)
    G.add_edge(int(u), int(v))

group_is_A = np.arange(N) < half        # True = A 陣營(攻擊傾向), False = B 陣營(護航傾向)
own_action = np.where(group_is_A, 1, -1)  # 各自的「極端行動」方向

# 每人的表態成本 C_i (公開門檻) 與基本陣營認同 priv_i (恆正，拉向自己的極端)
C = rng.beta(a5, b5, size=N)
priv = np.abs(rng.normal(PRIV_MEAN, PRIV_STD, size=N))

# KOL：每陣營各半，認同感極強、在「自己同溫層內」廣播 (放大回聲室)
kol_A = rng.choice(np.arange(half), size=NUM_KOLS // 2, replace=False)
kol_B = rng.choice(np.arange(half, N), size=NUM_KOLS // 2, replace=False)
for kol, lo, hi in [(k, 0, half) for k in kol_A] + [(k, half, N) for k in kol_B]:
    priv[kol] = 0.6
    targets = rng.integers(lo, hi, size=KOL_REACH)
    for t in targets:
        if kol != t:
            G.add_edge(int(kol), int(t))

A_adj = nx.to_scipy_sparse_array(G, nodelist=range(N), dtype=float, format="csr")
deg = np.asarray(A_adj.sum(axis=1)).ravel()
deg[deg == 0] = 1.0


# ---------------------------------------------------------------------------
# 2. 賽局核心
# ---------------------------------------------------------------------------
def run_polarization(kappa, deterministic, m=M_NOISE,
                     max_steps=40, avg_last=10, seed=0):
    """
    非單向：每步重新決策，跑到準穩態後取最後 avg_last 步平均。
    回傳 (polarization, mean_A, mean_B, history_pol)
      polarization = (mean_A − mean_B) / 2 ∈ [0,1]
      = 1 代表兩陣營各自完全動員到相反極端 (最大撕裂)
    """
    r = np.random.default_rng(seed)
    state = np.zeros(N)
    pol_hist = []

    for _ in range(max_steps):
        attack_active = (state == 1).astype(float)
        defend_active = (state == -1).astype(float)

        # 同溫層相挺：鄰居中跟我同立場的比例
        phi_in_A = (A_adj @ attack_active) / deg     # A 看同陣營攻擊者
        phi_in_B = (A_adj @ defend_active) / deg     # B 看同陣營護航者
        phi_in = np.where(group_is_A, phi_in_A, phi_in_B)

        # 全網敵營動員比例 (演算法 κ 決定我有多看得到他們)
        opp_global = np.where(group_is_A, defend_active.mean(), attack_active.mean())

        U = priv + B_IN * phi_in + B_REACT * kappa * opp_global - C - R

        if deterministic:
            commit = U >= 0
        else:
            commit = r.random(N) < 1.0 / (1.0 + np.exp(-m * U))

        state = np.where(commit, own_action, 0)

        mean_A = state[group_is_A].mean()
        mean_B = state[~group_is_A].mean()
        pol_hist.append((mean_A - mean_B) / 2)

    pol = float(np.mean(pol_hist[-avg_last:]))
    return pol, state[group_is_A].mean(), state[~group_is_A].mean(), pol_hist


# ---------------------------------------------------------------------------
# 3. 實驗一：掃描演算法強度 κ — 決定論 vs 雜訊的極化曲線
# ---------------------------------------------------------------------------
print(f"[{time.strftime('%H:%M:%S')}] 實驗一：掃描 κ (決定論 vs 雜訊)...")
kappa_range = np.linspace(0.0, 1.0, 16)
N_TRIALS = 5

pol_det, pol_noise = [], []
for kap in kappa_range:
    pol_det.append(run_polarization(kap, deterministic=True, seed=0)[0])
    trials = [run_polarization(kap, deterministic=False, m=M_NOISE, seed=s)[0]
              for s in range(N_TRIALS)]
    pol_noise.append(np.mean(trials))
    print(f"  κ={kap:.2f} | 決定論極化 {pol_det[-1]:.2f} | 雜訊極化 {pol_noise[-1]:.2f}")

# ---------------------------------------------------------------------------
# 4. 實驗二：在高 κ 下，極化隨時間如何演化
# ---------------------------------------------------------------------------
print(f"[{time.strftime('%H:%M:%S')}] 實驗二：高 κ=1.0 下的極化時間軌跡...")
_, _, _, hist_det = run_polarization(1.0, deterministic=True, seed=0)
_, _, _, hist_noise = run_polarization(1.0, deterministic=False, m=M_NOISE, seed=0)

print(f"[{time.strftime('%H:%M:%S')}] 完成，繪圖中...")

# ---------------------------------------------------------------------------
# 5. 繪圖
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# 圖 A：極化 vs 演算法強度 κ
ax1.plot(kappa_range, pol_det, "-o", color="black", lw=2.5,
         label="決定論")
ax1.plot(kappa_range, pol_noise, "-s", color="seagreen", lw=2.5,
         label=f"行為雜訊 (m={M_NOISE:.0f})")

ax1.set_title("演算法跨溫層強度 κ vs 社群極化程度", fontsize=13)
ax1.set_xlabel("演算法把敵營推給你的強度 κ")
ax1.set_ylabel("極化指數")
ax1.set_ylim(0, 1.05)
ax1.legend(loc="upper left")
ax1.grid(alpha=0.3)

# 圖 B：高 κ 下極化的時間軌跡
ax2.plot(hist_det, "-", color="black", lw=2.5, label="決定論")
ax2.plot(hist_noise, "-", color="seagreen", lw=2.5, label="加入雜訊")
ax2.set_title("高演算法強度 (κ=1.0) 下極化的時間演化", fontsize=13)
ax2.set_xlabel("Time Step (時間推移)")
ax2.set_ylabel("極化指數")
ax2.set_ylim(0, 1.0)
ax2.legend(loc="lower right")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("hawk_dove.png", dpi=130)
print(f"[{time.strftime('%H:%M:%S')}] 已存檔 hawk_dove.png")
plt.show()
