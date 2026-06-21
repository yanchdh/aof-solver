#!/usr/bin/env python3
"""
AOF (All-in or Fold) CFR+ Solver
================================
德州扑克翻前 All-in or Fold Nash 均衡求解器。

场景: 2-4 人, 可配置筹码深度(默认 8BB), 有效动作仅 Fold/All-in。
算法: CFR+ (Counterfactual Regret Minimization Plus)
手牌抽象: 169 类 (13 对子 + 78 同花 + 78 不同花)
权益计算:
  - 两人: 预计算 169x169 蒙特卡洛权益表
  - 多人: strength-weighted 近似 (手牌强度加权, 远优于 1/n 等分)

用法: python aof_cfr_solver.py [--players 4] [--stack 8] [--iters 10000]
"""

import random
import json
import os
import sys
import time
import itertools
import math
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

# ============================================================
# 0. 配置
# ============================================================

RANKS = '23456789TJQKA'
SUIT_CHARS = 'shdc'
N_HAND_TYPES = 169
N_PAIRS = 13         # 0-12: AA..22
N_SUITED = 78        # 13-90: AKs..32s
N_OFFSUIT = 78       # 91-168: AKo..32o
TOTAL_COMBOS = 1326  # C(52,2)

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1. 牌与手牌表示
# ============================================================

def card_idx(rank: int, suit: int) -> int:
    """rank 0-12 (2-A), suit 0-3 (s,h,d,c) -> 0-51"""
    return rank + suit * 13


def card_str(idx: int) -> str:
    return f"{RANKS[idx % 13]}{SUIT_CHARS[idx // 13]}"


def hand_str(hole: Tuple[int, int]) -> str:
    c1, c2 = hole
    return f"{card_str(c1)}{card_str(c2)}"


def hand_type_idx(c1: int, c2: int) -> int:
    """两张牌 -> 类型索引 0-168"""
    r1, r2 = c1 % 13, c2 % 13
    if r1 == r2:
        return 12 - r1  # AA=0, KK=1, ..., 22=12
    hi, lo = (r1, r2) if r1 > r2 else (r2, r1)
    is_suited = (c1 // 13 == c2 // 13)
    # suited 顺序: AKs(hi=12,lo=11) 开始, 按 hi 降序 lo 降序
    offset = 0
    for h in range(12, hi, -1):
        offset += h
    offset += lo
    if is_suited:
        return 13 + offset
    else:
        return 13 + 78 + offset


def type_str(t: int) -> str:
    """类型索引 -> 可读字符串"""
    if t <= 12:
        r = 12 - t
        return f"{RANKS[r]}{RANKS[r]}"
    idx = t - 13
    if idx < 78:
        off = idx
        for h in range(12, 0, -1):
            if off < h:
                return f"{RANKS[h]}{RANKS[off]}s"
            off -= h
        return f"{RANKS[1]}{RANKS[0]}s"
    else:
        off = idx - 78
        for h in range(12, 0, -1):
            if off < h:
                return f"{RANKS[h]}{RANKS[off]}o"
            off -= h


def type_combos(t: int) -> List[Tuple[int, int]]:
    """手牌类型的所有具体组合"""
    result = []
    if t <= 12:  # 口袋对: 6 种组合
        r = 12 - t
        for s1 in range(4):
            for s2 in range(s1 + 1, 4):
                result.append((r + s1 * 13, r + s2 * 13))
    else:
        idx = t - 13
        if idx < 78:
            off = idx
            hi = lo = None
            for h in range(12, 0, -1):
                if off < h:
                    hi, lo = h, off
                    break
                off -= h
            if hi is None:
                hi, lo = 1, 0
            for s in range(4):
                result.append((hi + s * 13, lo + s * 13))  # 4 种同花组合
        else:
            off = idx - 78
            hi = lo = None
            for h in range(12, 0, -1):
                if off < h:
                    hi, lo = h, off
                    break
                off -= h
            if hi is None:
                hi, lo = 1, 0
            for s1 in range(4):
                for s2 in range(4):
                    if s1 != s2:
                        result.append((hi + s1 * 13, lo + s2 * 13))  # 12 种不同花组合
    return result


def type_combo_count(t: int) -> int:
    if t <= 12:
        return 6
    idx = t - 13
    if idx < 78:
        return 4
    return 12


# 全局: combo 权重 (概率分布, sum = 1)
COMBO_WEIGHTS = [type_combo_count(t) / TOTAL_COMBOS for t in range(N_HAND_TYPES)]

# 所有 1326 手牌 -> 类型映射
ALL_HANDS = [(c1, c2) for c1 in range(51) for c2 in range(c1 + 1, 52)]
HAND_TO_TYPE = {h: hand_type_idx(*h) for h in ALL_HANDS}


# ============================================================
# 2. 手牌评估器 (简易 7 选 5)
# ============================================================

def evaluate_7cards(cards: List[int]) -> int:
    """从 7 张牌中选最佳 5 张, 返回评分 (越大越好)"""
    best = 0
    for combo in itertools.combinations(cards, 5):
        score = _evaluate_5cards(list(combo))
        if score > best:
            best = score
    return best


def _evaluate_5cards(cards: List[int]) -> int:
    """5 张牌的评分: [牌型 4bit][排名 4bit*5]"""
    ranks = sorted([c % 13 for c in cards], reverse=True)
    suits = [c // 13 for c in cards]
    is_flush = len(set(suits)) == 1
    is_straight = False
    straight_high = -1

    unique_ranks = sorted(set(ranks), reverse=True)
    for i in range(len(unique_ranks) - 4):
        if unique_ranks[i] - unique_ranks[i + 4] == 4:
            is_straight = True
            straight_high = unique_ranks[i]
            break
    if set(ranks) == {12, 0, 1, 2, 3}:
        is_straight = True
        straight_high = 3

    rank_counts = defaultdict(int)
    for r in ranks:
        rank_counts[r] += 1
    counts = sorted(rank_counts.values(), reverse=True)

    if is_straight and is_flush:
        return (8 << 20) | (straight_high << 16)
    if 4 in counts:
        quad = max(r for r, c in rank_counts.items() if c == 4)
        kick = max(r for r, c in rank_counts.items() if c != 4)
        return (7 << 20) | (quad << 16) | (kick << 12)
    if 3 in counts and 2 in counts:
        trip = max(r for r, c in rank_counts.items() if c == 3)
        pair = max(r for r, c in rank_counts.items() if c == 2)
        return (6 << 20) | (trip << 16) | (pair << 12)
    if is_flush:
        score = (5 << 20)
        for i, r in enumerate(ranks[:5]):
            score |= r << (16 - 4 * i)
        return score
    if is_straight:
        return (4 << 20) | (straight_high << 16)
    if 3 in counts:
        trip = max(r for r, c in rank_counts.items() if c == 3)
        kickers = sorted([r for r in ranks if r != trip], reverse=True)
        return (3 << 20) | (trip << 16) | (kickers[0] << 12) | (kickers[1] << 8)
    if counts.count(2) >= 2:
        pairs_ = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        kick = max(r for r in ranks if r not in pairs_[:2])
        return (2 << 20) | (pairs_[0] << 16) | (pairs_[1] << 12) | (kick << 8)
    if 2 in counts:
        pair = max(r for r, c in rank_counts.items() if c == 2)
        kickers = sorted([r for r in ranks if r != pair], reverse=True)
        return (1 << 20) | (pair << 16) | (kickers[0] << 12) | (kickers[1] << 8) | (kickers[2] << 4)
    score = 0
    for i, r in enumerate(ranks[:5]):
        score |= r << (16 - 4 * i)
    return score


# ============================================================
# 3. 蒙特卡洛权益计算
# ============================================================

def monte_carlo_equity(hole1: Tuple[int, int], hole2: Tuple[int, int],
                       n_iter: int = 2000) -> float:
    """计算 hole1 对 hole2 的 HU 权益 (win + tie/2)"""
    used = set(hole1) | set(hole2)
    deck = [c for c in range(52) if c not in used]
    wins, ties = 0, 0
    for _ in range(n_iter):
        random.shuffle(deck)
        board = deck[:5]
        score1 = evaluate_7cards(list(hole1) + board)
        score2 = evaluate_7cards(list(hole2) + board)
        if score1 > score2:
            wins += 1
        elif score1 == score2:
            ties += 1
    return (wins + ties / 2) / n_iter


def precompute_equity_table(n_iter: int = 2000, cache_path: str = None) -> List[List[float]]:
    """预计算 169x169 权益矩阵, 缓存到文件"""
    if cache_path is None:
        cache_path = os.path.join(CACHE_DIR, "equity_169.json")

    if os.path.exists(cache_path):
        print(f"[Equity] 加载缓存 {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    print(f"[Equity] 预计算 169x169 权益矩阵 ({n_iter} iter/pair)...")
    eq = [[0.0] * N_HAND_TYPES for _ in range(N_HAND_TYPES)]

    for i in range(N_HAND_TYPES):
        combos_i = type_combos(i)
        for j in range(i, N_HAND_TYPES):
            combos_j = type_combos(j)
            total_eq = 0.0
            total_n = 0
            # 对每个类型采样若干组合求平均
            sample_i = combos_i[:min(3, len(combos_i))]
            sample_j = combos_j[:min(3, len(combos_j))]
            for ci in sample_i:
                for cj in sample_j:
                    if len(set(ci) & set(cj)) > 0:
                        continue
                    eq_ij = monte_carlo_equity(ci, cj, n_iter)
                    total_eq += eq_ij
                    total_n += 1
            if total_n > 0:
                avg = total_eq / total_n
            else:
                avg = 0.5
            eq[i][j] = avg
            eq[j][i] = 1.0 - avg

        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/169 类型完成")
            # 增量保存
            with open(cache_path, 'w') as f:
                json.dump(eq, f)

    with open(cache_path, 'w') as f:
        json.dump(eq, f)
    print(f"[Equity] 矩阵已保存到 {cache_path}")
    return eq


def compute_strengths(eq_table: List[List[float]]) -> List[float]:
    """计算每类手牌的强度 = 对所有随机手牌的加权平均权益"""
    strengths = []
    for i in range(N_HAND_TYPES):
        s = sum(eq_table[i][j] * COMBO_WEIGHTS[j] for j in range(N_HAND_TYPES))
        strengths.append(s)
    return strengths


# ============================================================
# 4. AOF 游戏模型
# ============================================================

class AOFGame:
    """AOF 游戏定义"""
    def __init__(self, n_players: int, stack_bb: float):
        assert 2 <= n_players <= 4
        self.n = n_players
        self.stack = stack_bb
        # 盲注: 最后一名是 BB, 倒数第二是 SB
        self.blinds = [0.0] * n_players
        if n_players >= 2:
            self.blinds[-1] = 1.0   # BB
            self.blinds[-2] = 0.5   # SB
        self.behind = [stack_bb - b for b in self.blinds]

        # 位置名
        if n_players == 2:
            self.pos_names = ["SB", "BB"]
        elif n_players == 3:
            self.pos_names = ["BTN", "SB", "BB"]
        else:
            self.pos_names = ["UTG", "CO", "SB", "BB"]

        # 构建所有信息集
        self._build_info_sets()

    def _build_info_sets(self):
        """构建所有可能的 (player, history) 信息集"""
        self.histories = [[] for _ in range(self.n)]  # 每位玩家的历史列表
        self.history_idx = [{} for _ in range(self.n)]

        # 枚举所有前缀
        def enum_prefixes(p: int, prefix: str):
            if p >= self.n - 1:  # BB 之前
                return
            for a in 'FP':
                hist = prefix + a
                if p + 1 < self.n - 1:
                    enum_prefixes(p + 1, hist)
                else:
                    # SB 行动完, BB 可能不行动(全弃)
                    pass

        # UTG/BTN 的历史只有 ""
        self.histories[0] = [""]
        self.history_idx[0] = {"": 0}

        if self.n >= 3:
            # CO/SB 看到前一人行动
            self.histories[1] = ["F", "P"]
            self.history_idx[1] = {"F": 0, "P": 1}

        if self.n >= 4:
            # SB 看到前两人行动
            for a0 in 'FP':
                for a1 in 'FP':
                    h = a0 + a1
                    self.histories[2].append(h)
                    self.history_idx[2][h] = len(self.histories[2]) - 1

        # BB 的历史: 所有包含至少一次 Push 的前缀
        all_prefixes = []
        for bits in range(2 ** (self.n - 1)):
            prefix = ''.join('P' if (bits >> i) & 1 else 'F'
                             for i in range(self.n - 1))
            if 'P' in prefix:  # BB 只在有人 Push 后行动
                all_prefixes.append(prefix)

        self.histories[-1] = all_prefixes
        self.history_idx[-1] = {h: i for i, h in enumerate(all_prefixes)}

        print(f"[Game] {self.n}人, {self.stack}BB, 信息集: "
              + ", ".join(f"{self.pos_names[p]}:{len(self.histories[p])}"
                          for p in range(self.n)))

    def bb_acts_at(self, prefix: str) -> bool:
        """BB 是否需要在此前缀后行动"""
        return 'P' in prefix

    def pushers_before_bb(self, prefix: str) -> List[int]:
        """BB 之前推了的玩家"""
        return [i for i, a in enumerate(prefix) if a == 'P']


# 全局变量, 在 run() 中初始化
GAME: Optional[AOFGame] = None
EQ_TABLE: Optional[List[List[float]]] = None
STRENGTHS: Optional[List[float]] = None


# ============================================================
# 5. CFR+ 求解器
# ============================================================

class CFRSolver:
    def __init__(self, game: AOFGame):
        self.game = game
        self.n = game.n

        # 每个信息集存储一个 169 维向量的 regret (push 后悔减 fold 后悔)
        self.regret = [None] * self.n
        # 平均策略 (累积 push 概率和)
        self.strat_sum = [None] * self.n
        # 当前策略 (push 概率)
        self.strat = [None] * self.n

        for p in range(self.n):
            nh = len(game.histories[p])
            self.regret[p] = [[0.0] * N_HAND_TYPES for _ in range(nh)]
            self.strat_sum[p] = [[0.0] * N_HAND_TYPES for _ in range(nh)]
            # 初始策略: 50% push
            self.strat[p] = [[0.5] * N_HAND_TYPES for _ in range(nh)]

        # 收敛记录
        self.exploit_hist = []

    def get_strat(self, p: int, h_idx: int, i: int) -> float:
        return self.strat[p][h_idx][i]

    def _compute_bb_values(self, prefix: str):
        """
        计算 BB 在当前前缀下的 counterfactual value。
        返回 (v_fold[169], v_push[169], pots_info)
        """
        stack = self.game.stack
        pushers = self.game.pushers_before_bb(prefix)
        n_pushers = len(pushers)
        bb = self.n - 1
        nh = len(self.game.histories[bb])
        h_idx = self.game.history_idx[bb].get(prefix)
        if h_idx is None:
            return None, None, None

        # 底池计算
        pot_no_bb = sum(self.game.blinds[p] for p in range(self.n))
        for p in pushers:
            pot_no_bb += self.game.behind[p]

        pot_with_bb = pot_no_bb + self.game.behind[bb]

        v_fold = [-1.0] * N_HAND_TYPES  # BB 弃牌: 损失盲注

        if n_pushers == 0:
            # 不应该调用
            return v_fold, v_fold, (pot_no_bb, pot_with_bb)

        elif n_pushers == 1:
            # HU: BB vs 1 对手
            opp = pushers[0]
            # 对手的 push 范围权重
            opp_history = prefix[:opp]
            opp_h_idx = self.game.history_idx[opp][opp_history]
            opp_weights = [self.strat[opp][opp_h_idx][j] * COMBO_WEIGHTS[j]
                           for j in range(N_HAND_TYPES)]
            w_sum = sum(opp_weights)
            if w_sum < 1e-12:
                # 对手几乎不推 → BB push 赢盲注
                v_push = [(pot_no_bb - stack) * 169 for _ in range(N_HAND_TYPES)]
            else:
                v_push = [0.0] * N_HAND_TYPES
                for i in range(N_HAND_TYPES):
                    eq_i = sum(EQ_TABLE[i][j] * opp_weights[j] for j in range(N_HAND_TYPES)) / w_sum
                    v_push[i] = eq_i * pot_with_bb - stack

        else:
            # 多人: strength-weighted 近似
            # BB equity ≈ strength[i] / (strength[i] + Σ strength of opponents)
            opp_strength_sum = 0.0
            for opp in pushers:
                opp_history = prefix[:opp]
                opp_h_idx = self.game.history_idx[opp][opp_history]
                opp_strat = self.strat[opp][opp_h_idx]
                opp_strength = sum(STRENGTHS[j] * opp_strat[j] * COMBO_WEIGHTS[j]
                                   for j in range(N_HAND_TYPES))
                opp_weight_sum = sum(opp_strat[j] * COMBO_WEIGHTS[j] for j in range(N_HAND_TYPES))
                if opp_weight_sum > 1e-12:
                    opp_strength_sum += opp_strength / opp_weight_sum  # 加权平均强度

            v_push = [0.0] * N_HAND_TYPES
            for i in range(N_HAND_TYPES):
                eq_approx = STRENGTHS[i] / (STRENGTHS[i] + opp_strength_sum) if opp_strength_sum > 0 else 1.0
                v_push[i] = eq_approx * pot_with_bb - stack

        return v_fold, v_push, (pot_no_bb, pot_with_bb)

    def _update_bb(self, prefix: str, counterfactual_reach: float):
        """更新 BB 在此 prefix 下的后悔与策略和"""
        v_fold, v_push, _ = self._compute_bb_values(prefix)
        if v_fold is None:
            return

        bb = self.n - 1
        h_idx = self.game.history_idx[bb][prefix]

        for i in range(N_HAND_TYPES):
            push_prob = self.strat[bb][h_idx][i]
            ev = push_prob * v_push[i] + (1 - push_prob) * v_fold[i]

            reg_f = v_fold[i] - ev
            reg_p = v_push[i] - ev

            # CFR+ 累积 (非负 clamp)
            self.regret[bb][h_idx][i] = max(0.0,
                self.regret[bb][h_idx][i] + counterfactual_reach * (reg_p - reg_f))

            # 策略和累积 (用于平均策略)
            self.strat_sum[bb][h_idx][i] += counterfactual_reach * push_prob

    def _update_regret_matching(self):
        """对所有信息集执行 regret matching 策略更新"""
        for p in range(self.n):
            for h_idx in range(len(self.histories[p])):
                for i in range(N_HAND_TYPES):
                    # 将单值 regret 转为两个动作的 regret
                    r = self.regret[p][h_idx][i]
                    reg_push = max(r, 0.0)
                    reg_fold = max(-r, 0.0)
                    total = reg_push + reg_fold
                    if total > 1e-10:
                        self.strat[p][h_idx][i] = reg_push / total
                    else:
                        self.strat[p][h_idx][i] = 0.5

    def iterate(self, iter_num: int):
        """执行一次 CFR+ 迭代"""
        self._update_regret_matching()

        # 枚举所有 BB 行动前的前缀, 从 BB 视角更新
        # 对于 N 人, 枚举前 N-1 人的所有行动组合
        n_pre = self.n - 1

        for bits in range(2 ** n_pre):
            prefix = ''.join('P' if (bits >> i) & 1 else 'F'
                             for i in range(n_pre))
            if not self.game.bb_acts_at(prefix):
                continue  # 全弃, BB 不需要行动

            # 计算 counterfactual reach (对手到达此 prefix 的概率)
            reach = 1.0
            for p in range(n_pre):
                p_history = prefix[:p]
                p_h_idx = self.game.history_idx[p][p_history]
                action = prefix[p]
                # 平均策略概率
                avg_push = (sum(self.strat[p][p_h_idx][j] * COMBO_WEIGHTS[j]
                                for j in range(N_HAND_TYPES))
                            / sum(COMBO_WEIGHTS))
                if action == 'P':
                    reach *= max(avg_push, 0.001)
                else:
                    reach *= max(1.0 - avg_push, 0.001)

            # 更新 BB
            self._update_bb(prefix, reach)

            # --- 更新 SB（如果 SB 在这个 prefix 需要行动）---
            # SB 在 prefix[:n_pre-1] 处行动
            if self.n >= 3:
                sb = self.n - 2
                sb_history = prefix[:sb]
                sb_h_idx = self.game.history_idx[sb].get(sb_history)
                sb_action = prefix[sb] if len(prefix) > sb else None

                if sb_h_idx is not None and sb_action is not None:
                    # SB 的 counterfactual reach (除 SB 和 BB 外)
                    reach_sb = 1.0
                    for q in range(self.n):
                        if q == sb or q == self.n - 1:
                            continue
                        q_history = prefix[:q] if q < sb else prefix[:q]
                        if q >= len(prefix):
                            continue
                        q_h_idx = self.game.history_idx[q].get(prefix[:q])
                        if q_h_idx is None:
                            continue
                        avg_push_q = (sum(self.strat[q][q_h_idx][j] * COMBO_WEIGHTS[j]
                                          for j in range(N_HAND_TYPES))
                                      / sum(COMBO_WEIGHTS))
                        action_q = prefix[q]
                        if action_q == 'P':
                            reach_sb *= max(avg_push_q, 0.001)
                        else:
                            reach_sb *= max(1.0 - avg_push_q, 0.001)

                    self._update_sb(prefix, sb_h_idx, sb_action, reach_sb)

            # --- 更新 CO (如果有) ---
            if self.n >= 4:
                co = 1  # CO 是 index 1
                co_history = prefix[:co]
                co_h_idx = self.game.history_idx[co].get(co_history)
                co_action = prefix[co] if len(prefix) > co else None

                if co_h_idx is not None and co_action is not None:
                    reach_co = 1.0
                    for q in range(self.n):
                        if q == co or q >= 2:  # SB, BB
                            continue
                        if q >= len(prefix):
                            continue
                        q_h_idx = self.game.history_idx[q].get(prefix[:q])
                        if q_h_idx is None:
                            continue
                        avg_push_q = (sum(self.strat[q][q_h_idx][j] * COMBO_WEIGHTS[j]
                                          for j in range(N_HAND_TYPES))
                                      / sum(COMBO_WEIGHTS))
                        action_q = prefix[q]
                        if action_q == 'P':
                            reach_co *= max(avg_push_q, 0.001)
                        else:
                            reach_co *= max(1.0 - avg_push_q, 0.001)

                    self._update_co(prefix, co_h_idx, co_action, reach_co)

            # --- 更新 UTG ---
            utg = 0
            utg_action = prefix[0] if len(prefix) > 0 else None
            if utg_action is not None:
                # UTG 的 counterfactual reach: 后面所有玩家的到达概率
                reach_utg = 1.0
                for q in range(1, self.n):
                    q_history = prefix[:q]
                    q_h_idx = self.game.history_idx[q].get(q_history)
                    if q_h_idx is None:
                        continue
                    avg_push_q = (sum(self.strat[q][q_h_idx][j] * COMBO_WEIGHTS[j]
                                      for j in range(N_HAND_TYPES))
                                  / sum(COMBO_WEIGHTS))
                    if q < len(prefix):
                        action_q = prefix[q]
                        if action_q == 'P':
                            reach_utg *= max(avg_push_q, 0.001)
                        else:
                            reach_utg *= max(1.0 - avg_push_q, 0.001)

                self._update_early(0, prefix, reach_utg)

    def _update_sb(self, prefix: str, sb_h_idx: int, sb_action: str,
                   reach_sb: float):
        """更新 SB 的后悔"""
        sb = self.n - 2
        bb = self.n - 1
        stack = self.game.stack

        # SB 弃牌: 永远 -0.5
        v_fold = -0.5

        # SB 推: 取决于 BB 做什么
        # 构造 SB 推后的 prefix
        sb_prefix = prefix[:sb] + 'P'
        bb_history = sb_prefix
        bb_h_idx_after = self.game.history_idx[bb].get(bb_history)

        if bb_h_idx_after is None:
            # BB 不需要行动 (不可能, 因为 SB 推了)
            return

        # BB 弃牌概率
        bb_fold_prob = sum((1 - self.strat[bb][bb_h_idx_after][j]) * COMBO_WEIGHTS[j]
                           for j in range(N_HAND_TYPES)) / sum(COMBO_WEIGHTS)
        bb_call_prob = 1.0 - bb_fold_prob

        # 底池计算
        blind_total = sum(self.game.blinds)
        # SB 推前有多少人推了
        prev_pushers = [p for p, a in enumerate(prefix[:sb]) if a == 'P']
        pot_before_sb = blind_total + sum(self.game.behind[p] for p in prev_pushers)
        pot_after_sb = pot_before_sb + self.game.behind[sb]  # SB 推后, BB 弃牌时的底池
        pot_after_bb_call = pot_after_sb + self.game.behind[bb]

        # BB 弃牌: SB 收益
        profit_bb_fold = pot_after_sb - stack

        # BB 跟注: SB 权益
        for i in range(N_HAND_TYPES):
            # SB 对 BB 跟注范围的权益
            bb_call_weight = [(1.0 - self.strat[bb][bb_h_idx_after][j]) if False
                              else self.strat[bb][bb_h_idx_after][j] * COMBO_WEIGHTS[j]
                              for j in range(N_HAND_TYPES)]
            # 修正: BB call weight = bb_strat[j] * combo_weight[j]
            total_bb_w = sum(bb_call_weight)
            if total_bb_w < 1e-12:
                equity_vs_bb = 0.5
            else:
                equity_vs_bb = sum(EQ_TABLE[i][j] * bb_call_weight[j]
                                   for j in range(N_HAND_TYPES)) / total_bb_w

            v_push = bb_fold_prob * profit_bb_fold + bb_call_prob * (equity_vs_bb * pot_after_bb_call - stack)

            push_prob = self.strat[sb][sb_h_idx][i]
            ev = push_prob * v_push + (1 - push_prob) * v_fold

            reg_f = v_fold - ev
            reg_p = v_push - ev
            self.regret[sb][sb_h_idx][i] = max(0.0,
                self.regret[sb][sb_h_idx][i] + reach_sb * (reg_p - reg_f))
            self.strat_sum[sb][sb_h_idx][i] += reach_sb * push_prob

    def _update_co(self, prefix: str, co_h_idx: int, co_action: str,
                   reach_co: float):
        """更新 CO 的后悔 (仅 4 人)"""
        co = 1
        stack = self.game.stack
        v_fold = 0.0  # CO 没有盲注

        # CO 推后的 prefix
        co_prefix = prefix[:co] + 'P'
        # 后续 SB 和 BB 的行为会影响 CO 收益

        # SB 的策略
        sb = self.n - 2
        sb_history = co_prefix
        sb_h_idx_after = self.game.history_idx[sb].get(sb_history)
        if sb_h_idx_after is None:
            return

        sb_strat = self.strat[sb][sb_h_idx_after]
        sb_push_prob = sum(sb_strat[j] * COMBO_WEIGHTS[j] for j in range(N_HAND_TYPES)) / sum(COMBO_WEIGHTS)

        blind_total = sum(self.game.blinds)
        # CO 推前的 pushers
        prev_pushers = [p for p, a in enumerate(prefix[:co]) if a == 'P']
        pot_before_co = blind_total + sum(self.game.behind[p] for p in prev_pushers)
        pot_after_co = pot_before_co + self.game.behind[co]

        # 场景 1: SB 弃牌 → pot = pot_after_co + SB盲注, BB 决定
        # 场景 2: SB 推 → pot 更大, BB 决定

        # 简化: 用期望值
        bb = self.n - 1

        # SB 弃牌时
        bb_history_fold = co_prefix + 'F'
        bb_h_fold = self.game.history_idx[bb].get(bb_history_fold)

        # SB 推时
        bb_history_push = co_prefix + 'P'
        bb_h_push = self.game.history_idx[bb].get(bb_history_push)

        for i in range(N_HAND_TYPES):
            ev = 0.0

            # SB 弃牌 (概率 1-sb_push_prob)
            if bb_h_fold is not None:
                bb_strat_fold = self.strat[bb][bb_h_fold]
                bb_fold_p = (sum((1 - bb_strat_fold[j]) * COMBO_WEIGHTS[j] for j in range(N_HAND_TYPES))
                             / sum(COMBO_WEIGHTS))
                bb_call_p = 1.0 - bb_fold_p

                # pot: CO推 + SB盲(弃) + 前置贡献
                pot_no_bb = pot_after_co + self.game.blinds[sb]  # SB 弃牌贡献盲注
                pot_bb_call = pot_no_bb + self.game.behind[bb]

                # BB 弃牌: CO 赢
                profit_bb_fold = pot_no_bb - stack

                # BB 跟注: CO 权益 vs BB
                if bb_call_p > 0.001:
                    bb_call_w = [bb_strat_fold[j] * COMBO_WEIGHTS[j] for j in range(N_HAND_TYPES)]
                    w_sum = sum(bb_call_w)
                    if w_sum > 1e-12:
                        eq_vs_bb = sum(EQ_TABLE[i][j] * bb_call_w[j] for j in range(N_HAND_TYPES)) / w_sum
                    else:
                        eq_vs_bb = 0.5
                    ev_sb_fold = bb_fold_p * profit_bb_fold + bb_call_p * (eq_vs_bb * pot_bb_call - stack)
                else:
                    ev_sb_fold = profit_bb_fold

                ev += (1 - sb_push_prob) * ev_sb_fold

            # SB 推 (概率 sb_push_prob)
            if bb_h_push is not None and sb_push_prob > 0.001:
                bb_strat_push = self.strat[bb][bb_h_push]
                bb_fold_p = (sum((1 - bb_strat_push[j]) * COMBO_WEIGHTS[j] for j in range(N_HAND_TYPES))
                             / sum(COMBO_WEIGHTS))

                pot_with_sb = pot_before_co + self.game.behind[co] + self.game.behind[sb]
                pot_bb_call = pot_with_sb + self.game.behind[bb]

                # BB 弃牌: CO vs SB (或 CO 和 SB 都推, BB 弃)
                # 用 strength-weighted 近似多人
                co_str = STRENGTHS[i]
                sb_str = sum(STRENGTHS[j] * sb_strat[j] * COMBO_WEIGHTS[j] for j in range(N_HAND_TYPES))
                sb_w_sum = sum(sb_strat[j] * COMBO_WEIGHTS[j] for j in range(N_HAND_TYPES))
                if sb_w_sum > 1e-12:
                    sb_str /= sb_w_sum

                eq_vs_sb = co_str / (co_str + sb_str) if (co_str + sb_str) > 0 else 0.5

                # BB 跟注: CO vs SB vs BB
                bb_str = sum(STRENGTHS[j] * bb_strat_push[j] * COMBO_WEIGHTS[j] for j in range(N_HAND_TYPES))
                bb_w_sum = sum(bb_strat_push[j] * COMBO_WEIGHTS[j] for j in range(N_HAND_TYPES))
                if bb_w_sum > 1e-12:
                    bb_str /= bb_w_sum
                eq_vs_sb_bb = co_str / (co_str + sb_str + bb_str) if (co_str + sb_str + bb_str) > 0 else 0.33

                bb_call_p = 1.0 - bb_fold_p
                ev_sb_push = (bb_fold_p * (eq_vs_sb * pot_with_sb - stack)
                              + bb_call_p * (eq_vs_sb_bb * pot_bb_call - stack))
                ev += sb_push_prob * ev_sb_push

            # 更新后悔
            push_prob = self.strat[co][co_h_idx][i]
            current_ev = push_prob * ev + (1 - push_prob) * v_fold
            reg_f = v_fold - current_ev
            reg_p = ev - current_ev
            self.regret[co][co_h_idx][i] = max(0.0,
                self.regret[co][co_h_idx][i] + reach_co * (reg_p - reg_f))
            self.strat_sum[co][co_h_idx][i] += reach_co * push_prob

    def _update_early(self, p: int, prefix: str, reach: float):
        """更新早期位置 (UTG) 的后悔"""
        v_fold = -self.game.blinds[p]
        stack = self.game.stack
        h_idx = self.game.history_idx[p].get(prefix[:p] if p > 0 else "")
        if h_idx is None:
            return

        # p 推后的前缀
        push_prefix = prefix[:p] + 'P'
        action = prefix[p] if p < len(prefix) else None

        # 枚举后续所有可能的路径来计算期望收益
        total_ev = 0.0
        total_weight = 0.0

        # 对后续玩家的所有行动组合求期望
        n_remaining = self.n - p - 1
        if n_remaining == 0:
            return

        for bits in range(2 ** n_remaining):
            suffix = ''.join('P' if (bits >> q) & 1 else 'F' for q in range(n_remaining))
            full = push_prefix + suffix

            weight = 1.0
            pushers = [p]
            for q in range(n_remaining):
                next_p = p + 1 + q
                q_history = prefix[:next_p]
                if next_p < len(full):
                    q_hist = full[:next_p]
                    q_h_idx = self.game.history_idx[next_p].get(q_hist)
                    if q_h_idx is None:
                        weight = 0
                        break
                    q_strat = self.strat[next_p][q_h_idx]
                    action_q = suffix[q]
                    if action_q == 'P':
                        avg_push = sum(q_strat[j] * COMBO_WEIGHTS[j] for j in range(N_HAND_TYPES)) / sum(COMBO_WEIGHTS)
                        weight *= max(avg_push, 0.001)
                        pushers.append(next_p)
                    else:
                        avg_fold = sum((1 - q_strat[j]) * COMBO_WEIGHTS[j] for j in range(N_HAND_TYPES)) / sum(COMBO_WEIGHTS)
                        weight *= max(avg_fold, 0.001)
                else:
                    break

            if weight < 1e-10:
                continue

            # 计算此终端的收益 (简化: 只计算 p 的收益)
            blind_total = sum(self.game.blinds)
            pot = blind_total
            for pusher in pushers:
                pot += self.game.behind[pusher]

            n_pushers_total = len(pushers)
            # p 的近似权益 (strength-weighted)
            # 这里我们不做 per-hand 的精确计算，用平均强度估算
            # 真正 per-hand 的计算太昂贵
            # 使用平均收益来更新
            avg_profit = pot / n_pushers_total - stack if n_pushers_total > 0 else pot - stack

            total_ev += weight * avg_profit
            total_weight += weight

        if total_weight < 1e-10:
            return

        avg_ev = total_ev / total_weight

        # 对每个手牌，用强度调制收益
        for i in range(N_HAND_TYPES):
            # 强度调制: 强牌收益更高
            strength_factor = (STRENGTHS[i] - 0.5) * 2  # 映射到 [-1, 1]
            modulated_ev = avg_ev + strength_factor * 0.5 * abs(avg_ev - v_fold)

            v_push = modulated_ev

            push_prob = self.strat[p][h_idx][i]
            ev = push_prob * v_push + (1 - push_prob) * v_fold
            reg_f = v_fold - ev
            reg_p = v_push - ev
            self.regret[p][h_idx][i] = max(0.0,
                self.regret[p][h_idx][i] + reach * (reg_p - reg_f))
            self.strat_sum[p][h_idx][i] += reach * push_prob

    def compute_exploitability(self) -> float:
        """计算近似 exploitability (平均后悔值)"""
        total = 0.0
        n_el = 0
        for p in range(self.n):
            for h_idx in range(len(self.histories[p])):
                for i in range(N_HAND_TYPES):
                    total += abs(self.regret[p][h_idx][i])
                    n_el += 1
        return total / max(n_el, 1)

    def get_average_strategy(self, p: int, h_idx: int) -> List[float]:
        """返回某信息集的平均策略 (push 概率)"""
        strat = [0.0] * N_HAND_TYPES
        for i in range(N_HAND_TYPES):
            total = self.strat_sum[p][h_idx][i]
            # 标准化: 除以总迭代次数和 reach
            # 近似: 直接用累计值
            strat[i] = min(1.0, max(0.0, total / max(1.0, total)))
        return strat

    def compute_evs(self) -> Dict:
        """计算每个位置每种手牌的 EV"""
        results = {}
        for p in range(self.n):
            results[self.game.pos_names[p]] = {}
            for h_idx, history in enumerate(self.game.histories[p]):
                strat = self.get_average_strategy(p, h_idx)
                ev_list = [0.0] * N_HAND_TYPES

                # 使用当前策略计算近似 EV
                # 简化: EV = push_prob * avg_push_profit + (1-push_prob) * fold_profit
                for i in range(N_HAND_TYPES):
                    blind = self.game.blinds[p]
                    # 粗略 EV 估计
                    fold_ev = -blind
                    push_ev = STRENGTHS[i] * 2 - 0.5  # 简化: 强牌推有利
                    ev_list[i] = strat[i] * push_ev + (1 - strat[i]) * fold_ev

                key = history if history else "(开局)"
                results[self.game.pos_names[p]][key] = {
                    'strategies': strat,
                    'evs': ev_list
                }

        return results


# ============================================================
# 6. HTML 报告生成
# ============================================================

def generate_html(game: AOFGame, solver: CFRSolver, output_path: str):
    """生成 HTML 可视化报告"""
    avg_strats = {}
    for p in range(game.n):
        avg_strats[p] = {}
        for h_idx, history in enumerate(game.histories[p]):
            avg_strats[p][history if history else "(开局)"] = solver.get_average_strategy(p, h_idx)

    # 构建热力图矩阵: 13x13
    # 上三角: 同花 (行 < 列), 下三角: 不同花, 对角线: 对子

    def build_heatmap_data(strat: List[float]):
        """将 169 向量转为 13x13 热力图数据"""
        grid = [['' for _ in range(13)] for _ in range(13)]
        values = [[0.0 for _ in range(13)] for _ in range(13)]

        for row in range(13):
            # row 0 = A, row 12 = 2
            hi = 12 - row
            for col in range(13):
                lo = 12 - col
                if hi == lo:  # 对子
                    t_idx = 12 - hi
                    if 0 <= t_idx <= 12:
                        values[row][col] = strat[t_idx]
                elif hi > lo:  # 同花 (上三角: row < col)
                    offset = 0
                    for h in range(12, hi, -1):
                        offset += h
                    offset += lo
                    t_idx = 13 + offset
                    if t_idx < N_HAND_TYPES:
                        values[row][col] = strat[t_idx]
                else:  # hi < lo: 不同花 (下三角)
                    offset = 0
                    for h in range(12, lo, -1):
                        offset += h
                    offset += hi
                    t_idx = 13 + 78 + offset
                    if t_idx < N_HAND_TYPES:
                        values[row][col] = strat[t_idx]
                grid[row][col] = type_str(t_idx) if 't_idx' in dir() else '??'

        return grid, values

    html_parts = []

    for p in range(game.n):
        for hist_key, strat in avg_strats[p].items():
            grid, values = build_heatmap_data(strat)

            title = f"{game.pos_names[p]} — {hist_key}"
            hist_id = title.replace(" ", "_").replace("(", "").replace(")", "")

            html_parts.append(f"""
            <div class="range-section">
                <h3>{title}</h3>
                <table class="range-grid">
                    <tr><th></th>{"".join(f"<th>{RANKS[12-c]}</th>" for c in range(13))}</tr>
            """)

            for row in range(13):
                html_parts.append(f'<tr><th>{RANKS[12-row]}</th>')
                for col in range(13):
                    v = values[row][col]
                    # 绿=推, 红=弃
                    r = int(255 * (1 - v))
                    g = int(255 * v)
                    b = 0
                    html_parts.append(
                        f'<td style="background:rgb({r},{g},{b});color:{"#fff" if v > 0.5 else "#000"}">'
                        f'{v:.0%}</td>'
                    )
                html_parts.append('</tr>')

            html_parts.append('</table></div>')

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>AOF Push/Fold GTO - {game.n}人 {game.stack}BB</title>
<style>
body {{ font-family: 'Segoe UI',Arial,sans-serif; max-width:1200px; margin:0 auto; padding:20px; background:#1a1a2e; color:#eee; }}
h1 {{ color:#e94560; text-align:center; }}
h2 {{ color:#0f3460; background:#eee; padding:8px 16px; border-radius:4px; }}
h3 {{ color:#ccc; margin:20px 0 10px; }}
.range-section {{ margin:20px 0; padding:16px; background:#16213e; border-radius:8px; }}
.range-grid {{ border-collapse:collapse; font-size:12px; }}
.range-grid td, .range-grid th {{ width:36px; height:28px; text-align:center; border:1px solid #333; font-size:11px; }}
.range-grid th {{ background:#0f3460; color:#ccc; }}
.summary {{ background:#16213e; padding:16px; border-radius:8px; margin:20px 0; }}
.summary p {{ margin:4px 0; }}
</style>
</head>
<body>
<h1>🃏 AOF Push/Fold GTO 求解结果</h1>
<div class="summary">
    <p><strong>场景:</strong> {game.n} 人 All-in or Fold</p>
    <p><strong>筹码深度:</strong> {game.stack} BB</p>
    <p><strong>盲注:</strong> {" / ".join(f"{game.pos_names[p]}:{game.blinds[p]}BB" for p in range(game.n))}</p>
    <p><strong>算法:</strong> CFR+ (Counterfactual Regret Minimization Plus)</p>
    <p><strong>手牌抽象:</strong> 169 类</p>
    <p><strong>最终 exploitability:</strong> {solver.compute_exploitability():.6f}</p>
    <p><strong>说明:</strong> 绿色 = Push, 红色 = Fold, 百分比 = Push 概率</p>
</div>
<h2>Push / Fold 范围表</h2>
<p style="color:#999">上三角 = 同花, 下三角 = 不同花, 对角线 = 对子</p>
{"".join(html_parts)}
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[Report] 报告已生成: {output_path}")


# ============================================================
# 7. 主程序
# ============================================================

def main():
    global EQ_TABLE, STRENGTHS, GAME

    import argparse
    parser = argparse.ArgumentParser(description="AOF CFR+ Solver")
    parser.add_argument("--players", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--stack", type=float, default=8.0)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--equity-iters", type=int, default=2000,
                        help="权益预计算的蒙特卡洛迭代数")
    parser.add_argument("--no-cache", action="store_true",
                        help="重新计算权益表(忽略缓存)")
    parser.add_argument("--output", type=str, default=None,
                        help="HTML 输出路径")
    args = parser.parse_args()

    GAME = AOFGame(args.players, args.stack)

    # 预计算权益
    cache_path = os.path.join(CACHE_DIR, "equity_169.json") if not args.no_cache else None
    EQ_TABLE = precompute_equity_table(args.equity_iters, cache_path)
    STRENGTHS = compute_strengths(EQ_TABLE)

    # 显示权益表信息
    print(f"[Equity] AA vs 随机: {STRENGTHS[0]:.3f}")
    print(f"[Equity] AKs vs 随机: {STRENGTHS[13]:.3f}")
    print(f"[Equity] 72o vs 随机: {STRENGTHS[168]:.3f}")

    # CFR+ 求解
    solver = CFRSolver(GAME)
    print(f"\n[CFR+] 开始训练, {args.iters} 迭代...")
    start_time = time.time()

    report_interval = max(1, args.iters // 20)

    for it in range(1, args.iters + 1):
        solver.iterate(it)

        if it % report_interval == 0:
            exploit = solver.compute_exploitability()
            elapsed = time.time() - start_time
            print(f"  Iter {it:>6}/{args.iters} | exploit={exploit:.6f} | {elapsed:.1f}s")

    elapsed = time.time() - start_time
    exploit = solver.compute_exploitability()
    print(f"\n[Done] {args.iters} iters in {elapsed:.1f}s, final exploit={exploit:.6f}")

    # 输出 EV
    ev_results = solver.compute_evs()
    print("\n" + "=" * 60)
    print("EV 汇总 (BB)")
    print("=" * 60)

    for pos_name, histories in ev_results.items():
        print(f"\n--- {pos_name} ---")
        for hist_key, data in histories.items():
            push_range = [type_str(i) for i in range(N_HAND_TYPES) if data['strategies'][i] > 0.5]
            top_10_push = sorted(
                [(type_str(i), data['strategies'][i]) for i in range(N_HAND_TYPES)],
                key=lambda x: -x[1]
            )[:10]
            print(f"  {hist_key}:")
            print(f"    Top Push: {', '.join(f'{h}({p:.0%})' for h, p in top_10_push[:8])}")

    # 生成 HTML
    output_path = args.output or os.path.join(
        CACHE_DIR, f"aof_report_{args.players}p_{int(args.stack)}bb.html")
    generate_html(GAME, solver, output_path)
    return output_path


if __name__ == "__main__":
    output_file = main()
    print(f"\n✅ 完成! 报告: {output_file}")
