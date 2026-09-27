"""
取消文化：網路拓撲如何決定炎上 (連結密度 K / 重連率 P)

兩個經典文獻發現，直接對應你 baseline 裡的 K、P 兩個參數：

  圖 A. Watts (2002, PNAS) "A simple model of global cascades on random networks"
        -> 炎上規模對「平均連結度 K」是非單調的：
           太稀疏 (K 小) 沒有跨社群的連通路徑，火傳不出去；
           太稠密 (K 大) 每個鄰居只佔門檻的一小部分，誰都點不著。
           只有中間一段「級聯窗口 (cascade window)」會全面引爆。

  圖 B. Centola & Macy (2007) "Complex Contagions and the Weakness of Long Ties"
        -> 取消文化是需要「多方社會增強」的【複雜傳染】(要好幾個鄰居都在罵才跟進)，
           不是疾病那種一個來源就感染的【簡單傳染】。
           重連率 P 越高 (長距離弱連結越多)：
             - 簡單傳染：捷徑幫忙，擴散更快更廣 (Granovetter 弱連結的力量)。
             - 複雜傳染：長連結【破壞】局部鄰里的互相增強，擴散反而崩潰。

────────────────────────────────────────────────────────────────────────
為什麼這支檔【不】直接用問卷擬合的門檻？
  問卷門檻呈雙峰分布、且有大量接近 0 的「低門檻脆弱者」，會讓系統
  (1) 表現得像簡單傳染、洗掉 Watts 的上界窗口；
  (2) 變成刀刃般的相變 (B=1.8→5%、B=2.0→100%)，掃描只看得到階梯。
  那種雙穩態本身就是你問題定義裡的「臨界規模 tipping point」，已在
  其他檔展示。這裡為了把【拓撲】這個自變數獨立出來看，改用受控的
  固定比例門檻 φ* (Watts/Centola 原論文做法)，並關掉全域 κ 廣播與
  KOL 隨機長連結 (否則會繞過網路結構)。
────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import time

matplotlib.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Microsoft JhengHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ==========================================
# 參數設定
# ==========================================
N = 10_000
MAX_STEPS = 5000     # 純局部傳染在環狀晶格上是「波」，需要夠多步數才能擴滿
SEED_BLOCK = 30      # 在 ring 上點燃連續的一段節點 (彼此相鄰，形成火苗群聚)
                     # 複雜傳染需要鄰里互相增強，隨機撒種子是點不起來的

# 圖 A (Watts 窗口)：固定比例門檻，掃描 K
PHI_STAR_A = 0.40    # 需要 40% 的鄰居都在罵，我才跟進 (複雜傳染門檻)
P_FIXED_A = 0.10     # 掃 K 時固定的重連率

# 圖 B (Centola)：固定 K，掃描 P，比較簡單 vs 複雜傳染
K_FIXED_B = 8
PHI_STAR_COMPLEX = 0.25   # 複雜傳染：需要 25% (8 個鄰居中 ≥2 個) 才跟進
SIMPLE_COUNT = 1          # 簡單傳染：只要 ≥1 個鄰居就跟進 (像病毒)
# ==========================================

seed_block = np.arange(SEED_BLOCK)


def build_adjacency(k, p):
    """建立小世界網路，回傳對稱稀疏鄰接矩陣與 degree。"""
    k = int(round(k))
    if k % 2 == 1:           # watts_strogatz 需要偶數 k
        k += 1
    G = nx.watts_strogatz_graph(N, k=k, p=p, seed=42)
    A = nx.to_scipy_sparse_array(G, nodelist=range(N), format="csr", dtype=float)
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0      # 避免除以零
    return A, deg


def run_cascade(A, deg, mode="frac", phi_star=0.4, count_thr=1):
    """
    向量化單向級聯，回傳最終炎上比例。
      mode="frac"  : 比例門檻，當「發聲鄰居比例」>= phi_star 時跟進
      mode="count" : 絕對門檻，當「發聲鄰居數量」>= count_thr 時跟進 (簡單傳染)
    """
    state = np.zeros(N, dtype=float)
    state[seed_block] = 1.0

    for _ in range(MAX_STEPS):
        active_neighbors = A @ state
        if mode == "frac":
            trigger = (active_neighbors / deg) >= phi_star
        else:
            trigger = active_neighbors >= count_thr
        newly = (state == 0) & trigger
        if not newly.any():
            break
        state[newly] = 1.0   # 單向：一旦發聲就不回頭
    return state.mean()


# ---------------------------------------------------------------------------
# 實驗 A：掃描連結度 K → Watts 級聯窗口
# ---------------------------------------------------------------------------
print(f"[{time.strftime('%H:%M:%S')}] 實驗 A：掃描連結度 K (φ*={PHI_STAR_A}, P={P_FIXED_A})...")
k_values = np.array([2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 36, 42, 50, 60, 70, 80, 100])
finals_k = []
for k in k_values:
    A, deg = build_adjacency(k, P_FIXED_A)
    size = run_cascade(A, deg, mode="frac", phi_star=PHI_STAR_A)
    finals_k.append(size)
    print(f"  K={k:>3} -> 炎上規模 {size:.1%}")

# ---------------------------------------------------------------------------
# 實驗 B：掃描重連率 P → 簡單 vs 複雜傳染
# ---------------------------------------------------------------------------
print(f"[{time.strftime('%H:%M:%S')}] 實驗 B：掃描重連率 P (K={K_FIXED_B})...")
p_values = np.linspace(0.0, 0.6, 16)
finals_simple = []
finals_complex = []
for p in p_values:
    A, deg = build_adjacency(K_FIXED_B, p)
    s_simple = run_cascade(A, deg, mode="count", count_thr=SIMPLE_COUNT)
    s_complex = run_cascade(A, deg, mode="frac", phi_star=PHI_STAR_COMPLEX)
    finals_simple.append(s_simple)
    finals_complex.append(s_complex)
    print(f"  P={p:.2f} -> 簡單傳染 {s_simple:.1%} | 複雜傳染 {s_complex:.1%}")

print(f"[{time.strftime('%H:%M:%S')}] 完成，繪圖中...")

# ---------------------------------------------------------------------------
# 繪圖
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 圖 A：K 級聯窗口
finals_k = np.array(finals_k)
ax1.plot(k_values, finals_k, "-o", color="darkorange", lw=2.5, markersize=7)
ax1.set_title(f"A. 連結密度 K vs 炎上規模 (Watts 級聯窗口)\n(複雜傳染門檻 φ*={PHI_STAR_A}, P={P_FIXED_A})", fontsize=13)
ax1.set_xlabel("平均連結度 K (同溫層大小)", fontsize=11)
ax1.set_ylabel("最終炎上比例", fontsize=11)
ax1.set_ylim(-0.03, 1.05)
# 標出窗口
win = k_values[finals_k > 0.5]
if len(win) > 0:
    ax1.axvspan(win.min(), win.max(), color="orange", alpha=0.12)
    ax1.text((win.min()+win.max())/2, 0.5, "級聯窗口\n(全面炎上)",
             color="darkorange", ha="center", va="center", fontsize=10)
ax1.text(2, 0.08, "太稀疏\n傳不出去", color="gray", fontsize=9)
ax1.text(95, 0.08, "太稠密\n每個鄰居佔比太小", color="gray", fontsize=9, ha="right")
ax1.grid(alpha=0.3)

# 圖 B：P 簡單 vs 複雜傳染
ax2.plot(p_values, finals_simple, "-o", color="crimson", lw=2.5,
         label="簡單傳染 (≥1 鄰居即跟進，像病毒)")
ax2.plot(p_values, finals_complex, "-s", color="steelblue", lw=2.5,
         label=f"複雜傳染 (需 ≥{PHI_STAR_COMPLEX:.0%} 鄰居，像取消文化)")
ax2.set_title(f"B. 重連率 P vs 炎上規模 (弱連結的軟弱性)\n(K={K_FIXED_B})", fontsize=13)
ax2.set_xlabel("重連率 P (越右邊長距離弱連結越多)", fontsize=11)
ax2.set_ylabel("最終炎上比例", fontsize=11)
ax2.set_ylim(-0.03, 1.05)
ax2.annotate("捷徑幫忙擴散\n(弱連結的力量)", xy=(0.3, 1.0), xytext=(0.3, 0.7),
             color="crimson", fontsize=9, ha="center")
ax2.annotate("長連結破壞鄰里增強\n→ 複雜傳染崩潰", xy=(0.35, 0.02), xytext=(0.32, 0.35),
             color="steelblue", fontsize=9, ha="center",
             arrowprops=dict(arrowstyle="->", color="steelblue"))
ax2.legend(loc="center right", fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("network_topology.png", dpi=130)
print(f"[{time.strftime('%H:%M:%S')}] 已存檔 network_topology.png")
plt.show()
