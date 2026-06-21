"""
AOF Push/Fold 最佳范围求解器
==============================
迭代最佳响应法: 对每个位置/决策点, 找出使 EV 最大的 Push 范围。
输出 Push 范围热力图 & 每手牌 EV。

用法:
  python solve_aof.py --players 2  --stack 8  --iters 5000
  python solve_aof.py --players 4  --stack 8  --iters 5000
"""

import json
import os
import sys
import time
import math
from typing import List, Dict, Tuple, Optional

# ============================================================
# 配置 / 常量
# ============================================================

RANKS = '23456789TJQKA'
N_TYPES = 169  # 13对子 + 78同花 + 78不同花
TOTAL_COMBOS = 1326
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "equity_169.json")


# ============================================================
# 手牌工具函数
# ============================================================

def type_str(t: int) -> str:
    if t <= 12: return f"{RANKS[12-t]}{RANKS[12-t]}"
    idx = t - 13
    if idx < 78:
        off = idx
        for h in range(12, 0, -1):
            if off < h: return f"{RANKS[h]}{RANKS[off]}s"
            off -= h
        return f"{RANKS[1]}{RANKS[0]}s"
    else:
        off = idx - 78
        for h in range(12, 0, -1):
            if off < h: return f"{RANKS[h]}{RANKS[off]}o"
            off -= h
        return f"{RANKS[1]}{RANKS[0]}o"


def type_combo_count(t: int) -> int:
    return 6 if t <= 12 else (4 if (t - 13) < 78 else 12)


COMBO_WEIGHTS = [type_combo_count(t) / TOTAL_COMBOS for t in range(N_TYPES)]


# ============================================================
# 加载权益表
# ============================================================

def load_equity() -> Tuple[List[List[float]], List[float]]:
    if not os.path.exists(CACHE):
        print("错误: 权益表未生成, 请先运行 precompute_equity.py")
        sys.exit(1)
    with open(CACHE) as f:
        eq = json.load(f)

    # 计算每手牌的 "强度" = 对随机手的权益
    strengths = [0.0] * N_TYPES
    for i in range(N_TYPES):
        s = 0.0
        for j in range(N_TYPES):
            s += eq[i][j] * COMBO_WEIGHTS[j]
        strengths[i] = s

    return eq, strengths


# ============================================================
# 游戏定义
# ============================================================

class Game:
    def __init__(self, n: int, stack: float, rake: float = 0.0):
        """
        n: 玩家数
        stack: 有效筹码 (BB)
        rake: 每手固定抽水 (BB), GG Poker AoF 通常 0.07-0.20
        """
        self.n = n
        self.effective = stack  # 有效筹码
        self.rake = rake        # 每手固定抽水 (BB)

        # 总筹码 (BB 多 1BB)
        self.total_stacks = [stack] * n
        self.total_stacks[-1] = stack + 1.0  # BB

        # 盲注
        self.blinds = [0.0] * n
        self.blinds[-1] = 1.0
        if n >= 2:
            self.blinds[-2] = 0.5

        # 有效贡献: BB 最多投入 effective_stack
        self.contribution = [min(self.total_stacks[p], self.effective) for p in range(n)]

        # 后手 = 有效贡献 - 已下盲注 (推牌时额外投入的量)
        self.behind = [self.contribution[p] - self.blinds[p] for p in range(n)]

        if n == 2:
            self.names = ["SB", "BB"]
        elif n == 3:
            self.names = ["BTN", "SB", "BB"]
        else:
            self.names = ["UTG", "BTN", "SB", "BB"]

        # 构建所有 (player, history) 信息集
        self._build()

    def _build(self):
        """构建信息集列表"""
        # 每个玩家在所有前缀下的历史
        hs = []
        for p in range(self.n):
            histories = []
            idx_map = {}
            # 枚举前 p 个玩家的所有行动 (不包含全 F)
            if p == 0:
                histories = [""]
                idx_map = {"": 0}
            elif p == self.n - 1:  # BB
                # 前 n-1 人的行动组合, 包含至少一个 P
                for bits in range(2 ** (self.n - 1)):
                    prefix = ''.join('P' if (bits >> k) & 1 else 'F'
                                     for k in range(self.n - 1))
                    if 'P' in prefix:
                        idx_map[prefix] = len(histories)
                        histories.append(prefix)
            else:
                for bits in range(2 ** p):
                    prefix = ''.join('P' if (bits >> k) & 1 else 'F'
                                     for k in range(p))
                    idx_map[prefix] = len(histories)
                    histories.append(prefix)
            hs.append((histories, idx_map))

        self.histories = [h[0] for h in hs]
        self.h_idx = [h[1] for h in hs]

        # 枚举所有完整路径 (前 n-1 人), 用于最佳响应计算
        # BB 路径 = 前 n-1 人至少一个 P
        self.all_paths = []
        for bits in range(2 ** (self.n - 1)):
            path = ''.join('P' if (bits >> k) & 1 else 'F' for k in range(self.n - 1))
            if 'P' in path:
                self.all_paths.append(path)

        print(f"[Game] {self.n}人 eff={self.effective}BB BB={self.total_stacks[-1]}BB rake={self.rake}BB: "
              + ", ".join(f"{self.names[p]}:{len(self.histories[p])}info"
                          for p in range(self.n)))


# ============================================================
# 策略表示 & 更新
# ============================================================

class Strategies:
    """所有信息集的策略: strategy[p][h_idx][hand_i] = push 概率"""

    def __init__(self, game: Game):
        self.game = game
        self.n = game.n
        # 初始: 根据位置推前 N% 的牌
        self.arr = []
        for p in range(game.n):
            nh = len(game.histories[p])
            mat = [[0.0] * N_TYPES for _ in range(nh)]
            for h_idx in range(nh):
                for i in range(N_TYPES):
                    mat[h_idx][i] = 0.5  # 初始 50%
            self.arr.append(mat)

    def get(self, p: int, h: str) -> List[float]:
        """获取玩家 p 在历史 h 下的策略向量 (169 维)"""
        return self.arr[p][self.game.h_idx[p][h]]

    def set_smoothed(self, p: int, h: str, br: List[float], alpha: float = 0.3):
        """用平滑更新: new = (1-alpha)*old + alpha*best_response"""
        old = self.get(p, h)
        hi = self.game.h_idx[p][h]
        for i in range(N_TYPES):
            self.arr[p][hi][i] = (1.0 - alpha) * self.arr[p][hi][i] + alpha * br[i]

    def avg_push_prob(self, p: int, h: str) -> float:
        """玩家 p 在 h 下 Push 的平均概率"""
        s = self.get(p, h)
        total = sum(s[i] * COMBO_WEIGHTS[i] for i in range(N_TYPES))
        return total / sum(COMBO_WEIGHTS)

    def avg_strength(self, p: int, h: str, strengths: List[float]) -> float:
        """玩家 p 在 h 下 Push 范围的平均手牌强度"""
        s = self.get(p, h)
        total_s = 0.0
        total_w = 0.0
        for i in range(N_TYPES):
            w = s[i] * COMBO_WEIGHTS[i]
            total_s += strengths[i] * w
            total_w += w
        return total_s / total_w if total_w > 1e-10 else 0.5

    def equity_vs_range(self, hand_i: int, opp_p: int, opp_h: str,
                        eq_table: List[List[float]]) -> float:
        """手牌 i 对对手 opp_p 在 opp_h 下的 Push 范围的权益"""
        opp_s = self.get(opp_p, opp_h)
        total = 0.0
        total_w = 0.0
        for j in range(N_TYPES):
            w = opp_s[j] * COMBO_WEIGHTS[j]
            total += eq_table[hand_i][j] * w
            total_w += w
        return total / total_w if total_w > 1e-10 else 0.5


# ============================================================
# 最佳响应计算 (核心)
# ============================================================

def compute_best_response(game: Game, p: int, h: str,
                          strats: Strategies,
                          eq_table: List[List[float]],
                          strengths: List[float],
                          jp_ev: List[float]):
    """
    计算玩家 p 在历史 h 下的最佳响应。
    返回: (push_ev[169], fold_ev[169], best_response[169])
    其中 best_response[i] = 1.0 if push_ev[i] > fold_ev[i] else 0.0
    """
    n = game.n
    contrib_p = game.contribution[p]  # 玩家 p 的有效投入
    v_fold = -game.blinds[p]  # 弃牌总是输盲注

    v_push = [0.0] * N_TYPES

    # ─── 确定 p 之前的推者 ───
    pre_pushers = [q for q, a in enumerate(h) if a == 'P']

    # ─── 确定 p 之后需要行动的玩家 ───
    post_players = list(range(p + 1, n))

    if not post_players:
        # p 是最后一个行动的 (BB): pot = total blinds + behind of all pushers
        total_blinds = sum(game.blinds)
        pot = total_blinds
        for q in pre_pushers + [p]:
            pot += game.behind[q]
        pot -= game.rake  # 抽水

        n_opp = len(pre_pushers)
        if n_opp == 0:
            pass
        elif n_opp == 1:
            opp = pre_pushers[0]
            opp_h = h[:opp]
            for i in range(N_TYPES):
                eq_i = strats.equity_vs_range(i, opp, opp_h, eq_table)
                v_push[i] = eq_i * pot - contrib_p + jp_ev[i]
        else:
            # 多人
            opp_strengths = []
            for opp in pre_pushers:
                opp_h = h[:opp]
                opp_strengths.append(strats.avg_strength(opp, opp_h, strengths))
            for i in range(N_TYPES):
                s_sum = strengths[i] + sum(opp_strengths)
                eq_i = strengths[i] / s_sum if s_sum > 0 else 1.0 / (n_opp + 1)
                v_push[i] = eq_i * pot - contrib_p + jp_ev[i]

    else:
        # p 不是最后一个, 需要枚举后续行动
        n_post = len(post_players)
        for bits in range(2 ** n_post):
            suffix = ''.join('P' if (bits >> k) & 1 else 'F'
                             for k in range(n_post))

            weight = 1.0
            all_pushers = list(pre_pushers) + [p]

            # 计算权重和最终推者集合
            skip = False
            for k, q in enumerate(post_players):
                q_h = h + 'P' + suffix[:k]
                # q 只看得到 p 推 + 之前的后缀
                if suffix[k] == 'P':
                    all_pushers.append(q)
                    prob = strats.avg_push_prob(q, q_h)
                else:
                    prob = 1.0 - strats.avg_push_prob(q, q_h)
                if prob < 1e-6:
                    skip = True
                    break
                weight *= prob
            if skip:
                continue

            # 计算此终端的底池
            total_blinds = sum(game.blinds)
            pot = total_blinds - game.rake  # 抽水
            for pusher in all_pushers:
                pot += game.behind[pusher]

            n_pushers = len(all_pushers)

            if n_pushers == 1:
                # 只有 p 推, 其他人弃 → 赢盲注
                profit = pot - contrib_p
                for i in range(N_TYPES):
                    v_push[i] += weight * profit
            elif n_pushers == 2:
                # p vs 1 对手
                opp = [q for q in all_pushers if q != p][0]
                # 对手的历史: 其看见的前缀
                if opp < p:
                    opp_h = h[:opp]
                else:
                    k_in_post = opp - p - 1
                    opp_h = h + 'P' + suffix[:k_in_post]

                for i in range(N_TYPES):
                    eq_i = strats.equity_vs_range(i, opp, opp_h, eq_table)
                    profit = eq_i * pot - contrib_p
                    v_push[i] += weight * profit
            else:
                # 多人
                opp_strengths = []
                for opp in all_pushers:
                    if opp == p:
                        continue
                    if opp < p:
                        opp_h = h[:opp]
                    else:
                        k = opp - p - 1
                        opp_h = h + 'P' + suffix[:k]
                    opp_strengths.append(strats.avg_strength(opp, opp_h, strengths))

                for i in range(N_TYPES):
                    s_sum = strengths[i] + sum(opp_strengths)
                    eq_i = strengths[i] / s_sum if s_sum > 0 else 1.0 / n_pushers
                    profit = eq_i * pot - contrib_p
                    v_push[i] += weight * profit

    # 最佳响应: 推 if push_ev > fold_ev
    br = [0.0] * N_TYPES
    for i in range(N_TYPES):
        br[i] = 1.0 if v_push[i] > v_fold else 0.0

    return v_push, [v_fold] * N_TYPES, br


# ============================================================
# 主迭代循环
# ============================================================

def solve(game: Game, eq_table: List[List[float]], strengths: List[float],
          jp_ev: List[float], n_iters: int = 5000, alpha: float = 0.2) -> Strategies:
    """
    迭代最佳响应求解。
    每轮从 BB → UTG 更新每个信息集。
    """
    strats = Strategies(game)
    print(f"\n[求解] {n_iters} 迭代, 平滑 α={alpha}")

    # 手牌排名 (按强度从强到弱排序), 用于初始化
    ranked = sorted(range(N_TYPES), key=lambda i: -strengths[i])

    # 初始策略: 按位置给初始 Push 范围
    init_ranges = {2: [0.55, 0.40], 3: [0.45, 0.35, 0.30], 4: [0.35, 0.30, 0.25, 0.22]}
    top_pct = init_ranges.get(game.n, [0.3] * game.n)
    for p in range(game.n):
        for h_idx in range(len(game.histories[p])):
            cutoff = int(top_pct[p] * N_TYPES)
            for i in range(N_TYPES):
                if ranked[i] in ranked[:cutoff]:
                    strats.arr[p][h_idx][ranked[i]] = 1.0
                else:
                    strats.arr[p][h_idx][ranked[i]] = 0.0

    start = time.time()
    for it in range(1, n_iters + 1):
        max_change = 0.0

        # 从后往前更新
        for p in range(game.n - 1, -1, -1):
            for h in game.histories[p]:
                v_push, v_fold, br = compute_best_response(
                    game, p, h, strats, eq_table, strengths, jp_ev)

                # 计算变化
                old = strats.get(p, h)
                change = sum(abs(br[i] - old[i]) * COMBO_WEIGHTS[i]
                             for i in range(N_TYPES))
                max_change = max(max_change, change)

                # 平滑更新
                strats.set_smoothed(p, h, br, alpha)

        if it % max(1, n_iters // 20) == 0:
            elapsed = time.time() - start
            print(f"  iter {it:>6}/{n_iters} | max_change={max_change:.4f} | {elapsed:.1f}s")

        if max_change < 0.0005 and it > 500:
            print(f"  [收敛] iter {it}, change={max_change:.6f}")
            break

    elapsed = time.time() - start
    print(f"[完成] {elapsed:.1f}s")
    return strats


# ============================================================
# 输出
# ============================================================

def print_results(game: Game, strats: Strategies,
                  eq_table: List[List[float]], strengths: List[float],
                  jp_ev: List[float]):
    """打印文本结果"""

    print("\n" + "=" * 70)
    print(f"  AOF {game.n}人 {game.effective}BB — Push 范围 & EV")
    print("=" * 70)

    for p in range(game.n):
        print(f"\n{'─' * 50}")
        print(f"  [{game.names[p]}]")
        for h in game.histories[p]:
            label = h if h else "(开局)"
            print(f"\n  决策点: {label}")

            s = strats.get(p, h)
            v_push, v_fold, br = compute_best_response(
                game, p, h, strats, eq_table, strengths, jp_ev)

            # 推的牌 (push_ev > fold_ev) 按强度排序
            push_hands = [(i, strengths[i], v_push[i])
                          for i in range(N_TYPES) if v_push[i] > v_fold[i]]
            push_hands.sort(key=lambda x: -x[1])

            # Top push 手牌
            top = push_hands[:12]
            print(f"  推荐 Push ({len(push_hands)}/{N_TYPES} 类):")
            lines = []
            for t, s_val, ev in top:
                lines.append(f"{type_str(t)}(EV={ev:+.2f})")
                if len(lines) == 6:
                    print(f"    " + "  ".join(lines))
                    lines = []
            if lines:
                print(f"    " + "  ".join(lines))

            # 关键边界
            if push_hands:
                print(f"  最弱推牌: {type_str(push_hands[-1][0])} "
                      f"(EV_push={push_hands[-1][2]:+.2f}, EV_fold={v_fold[0]:+.1f})")

            # BB 跟注范围 (如果存在)
            if p == game.n - 1:
                call_range = [i for i in range(N_TYPES) if v_push[i] > -1.0]
                call_range.sort(key=lambda i: -strengths[i])
                top_call = [type_str(i) for i in call_range[:8]]
                print(f"  Call 范围: {len(call_range)} 类 → {', '.join(top_call)}")


def generate_html(game: Game, strats: Strategies,
                  eq_table: List[List[float]], strengths: List[float],
                  jp_ev: List[float], out_path: str):
    """生成 HTML 热力图报告"""

    def _build_grid(strat_vec):
        """169 向量 → 13x13 (values, labels)"""
        vals = [[0.0]*13 for _ in range(13)]
        lbs = [[""]*13 for _ in range(13)]
        for row in range(13):
            hi = 12 - row
            for col in range(13):
                lo = 12 - col
                if hi == lo:
                    t = 12 - hi
                    suff = ""
                elif hi > lo:
                    off = 0
                    for hh in range(12, hi, -1): off += hh
                    t = 13 + off + lo
                    suff = "s"
                else:
                    off = 0
                    for hh in range(12, lo, -1): off += hh
                    t = 13 + 78 + off + hi
                    suff = "o"
                if 0 <= t < N_TYPES:
                    vals[row][col] = strat_vec[t]
                    r1, r2 = RANKS[hi], RANKS[lo]
                    lbs[row][col] = f"{r1}{r2}{suff}"
        return vals, lbs

    sections = []
    for p in range(game.n):
        for h in game.histories[p]:
            label = h if h else "(开局)"
            title = f"{game.names[p]} — {label}"
            s = strats.get(p, h)
            v_push, v_fold, br = compute_best_response(
                game, p, h, strats, eq_table, strengths, jp_ev)
            grid, lbs = _build_grid(s)

            # 推牌 EV
            push_hands = [(type_str(i), v_push[i], v_fold[i], strengths[i])
                          for i in range(N_TYPES) if v_push[i] > v_fold[i]]
            push_hands.sort(key=lambda x: -x[1])

            rows = []
            for r in range(13):
                cells = f'<th>{RANKS[12-r]}</th>'
                for c in range(13):
                    v = grid[r][c]
                    red = int(255 * (1 - v))
                    grn = int(255 * v)
                    color = f"rgb({red},{grn},0)"
                    text_color = "#fff" if v > 0.5 else "#111"

                    # 区分同花/不同花/对子的边框
                    if r < c:    border = "2px solid #4a90d9"  # 上三角=同花 蓝边框
                    elif r > c:  border = "2px solid #d98a4a"  # 下三角=不同花 橙边框
                    else:        border = "2px solid #fff"     # 对角线=对子 白边框

                    # 显示手牌名 + 同花/不同花标记
                    lbl = lbs[r][c]
                    cells += (f'<td style="background:{color};color:{text_color};border:{border};font-size:10px"'
                              f' title="{lbl}">{lbl}<br>{v:.0%}</td>')
                rows.append(f'<tr>{cells}</tr>')

            push_str = "<br>".join(
                f"{h}(EV={p:+4.1f})" for h,p,f,s in push_hands[:10]) if push_hands else "无"

            sections.append(f"""
            <div class="block">
              <h3>{title}</h3>
              <div style="display:flex;gap:20px;flex-wrap:wrap">
                <div>
                  <table class="grid">
                    <tr><th></th>{''.join(f'<th>{RANKS[12-c]}</th>' for c in range(13))}</tr>
                    {''.join(rows)}
                  </table>
                  <p style="color:#ccc;font-size:12px;margin:8px 0">
                    <span style="border:2px solid #4a90d9;padding:2px 6px;border-radius:3px;margin-right:8px">蓝框=同花(s)</span>
                    <span style="border:2px solid #d98a4a;padding:2px 6px;border-radius:3px;margin-right:8px">橙框=不同花(o)</span>
                    <span style="border:2px solid #fff;padding:2px 6px;border-radius:3px">白框=对子</span>
                  </p>
                </div>
                <div style="font-size:12px;min-width:180px">
                  <b>Top 推牌 (EV):</b><br>{push_str}
                </div>
              </div>
            </div>""")

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>AOF {game.n}人 {game.effective}BB 求解</title>
<style>
body{{font-family:'Segoe UI',Arial;max-width:1300px;margin:0 auto;padding:20px;
      background:#1a1a2e;color:#ddd}}
h1{{color:#e94560;text-align:center}}
h3{{color:#ccc;margin:16px 0 8px;border-left:3px solid #e94560;padding-left:8px}}
.block{{background:#16213e;border-radius:8px;padding:16px;margin:16px 0}}
.grid{{border-collapse:collapse;font-size:11px}}
.grid td,.grid th{{width:32px;height:24px;text-align:center;border:1px solid #333}}
.grid th{{background:#0f3460;color:#aaa}}
.summary{{background:#16213e;padding:16px;border-radius:8px;margin:16px 0}}
</style></head><body>
<h1>🃏 AOF Push/Fold — {game.n}人 {game.effective}BB</h1>
<div class="summary">
  <p>位置: {' → '.join(game.names)} &nbsp;|&nbsp;
  盲注: {', '.join(f'{n}:{b}BB' for n,b in zip(game.names,game.blinds))}</p>
  <p>绿色=推 红色=弃 &nbsp;|&nbsp; 每格显示手牌名(如AKs=同花, AKo=不同花)和推牌概率</p>
</div>
{"".join(sections)}
</body></html>"""

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n[报告] {out_path}")
    return out_path


# ============================================================
# 主入口
# ============================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(description="AOF Push/Fold 范围求解")
    ap.add_argument("--players", type=int, default=4, choices=[2, 3, 4])
    ap.add_argument("--stack", type=float, default=8.0)
    ap.add_argument("--rake", type=float, default=0.0,
                    help="每手固定抽水 (BB)")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--alpha", type=float, default=0.2,
                    help="策略更新平滑系数")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--json", type=str, default=None,
                    help="Export Nash strategies as JSON")
    args = ap.parse_args()

    eq_table, strengths = load_equity()
    print(f"[权益] AA={strengths[0]:.3f} AKs(24)={strengths[24]:.3f} 72o(154)={strengths[154]:.3f}")

    # Jackpot EV
    jp_path = os.path.join(HERE, "jackpot_ev.json")
    if os.path.exists(jp_path):
        with open(jp_path) as f:
            jp_ev = json.load(f)
        print(f"[Jackpot] 76s={jp_ev[80]:.2f}BB JTs={jp_ev[54]:.2f}BB 54s={jp_ev[87]:.2f}BB")
    else:
        jp_ev = [0.0] * N_TYPES

    game = Game(args.players, args.stack, args.rake)
    strats = solve(game, eq_table, strengths, jp_ev, args.iters, args.alpha)

    print_results(game, strats, eq_table, strengths, jp_ev)

    # Export JSON
    json_out = args.json
    if json_out:
        export = {
            "n": game.n,
            "effective": game.effective,
            "rake": game.rake,
            "names": game.names,
            "blinds": game.blinds,
            "total_stacks": game.total_stacks,
            "histories": game.histories,
            "strategies": strats.arr
        }
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump(export, f)
        print(f"[JSON] {json_out}")

    out = args.output or os.path.join(
        HERE, f"aof_{args.players}p_{int(args.stack)}bb.html")
    generate_html(game, strats, eq_table, strengths, jp_ev, out)
    return out


if __name__ == "__main__":
    out = main()
    print(f"\n[OK] Report: {out}")
