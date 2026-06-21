"""
快速权益预计算 — 并行版
使用 multiprocessing 并行计算 Monte Carlo
"""

import random
import json
import itertools
import os
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count

RANKS = '23456789TJQKA'
N_TYPES = 169
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DEFAULT = os.path.join(HERE, "equity_169.json")
CACHE = os.environ.get("EQUITY_CACHE", CACHE_DEFAULT)


def type_combos(t):
    r = []
    if t <= 12:
        rank = 12 - t
        for s1 in range(4):
            for s2 in range(s1+1, 4):
                r.append((rank + s1*13, rank + s2*13))
    else:
        idx = t - 13
        if idx < 78:
            off = idx; hi = lo = None
            for h in range(12, 0, -1):
                if off < h: hi, lo = h, off; break
                off -= h
            if hi is None: hi, lo = 1, 0
            for s in range(4):
                r.append((hi + s*13, lo + s*13))
        else:
            off = idx - 78; hi = lo = None
            for h in range(12, 0, -1):
                if off < h: hi, lo = h, off; break
                off -= h
            if hi is None: hi, lo = 1, 0
            for s1 in range(4):
                for s2 in range(4):
                    if s1 != s2:
                        r.append((hi + s1*13, lo + s2*13))
    return r


def type_combo_count(t):
    return 6 if t <= 12 else (4 if (t - 13) < 78 else 12)


# ===== Fast 7-card evaluator =====
def _score_5(ranks, suits):
    is_flush = len(set(suits)) == 1
    ur = sorted(set(ranks), reverse=True)
    straight_hi = -1
    for i in range(len(ur) - 4):
        if ur[i] - ur[i+4] == 4:
            straight_hi = ur[i]; break
    if set(ur) & {12, 0, 1, 2, 3} == {12, 0, 1, 2, 3}:
        straight_hi = 3
    rc = defaultdict(int)
    for r in ranks: rc[r] += 1
    cnt = sorted(rc.values(), reverse=True)
    if straight_hi >= 0 and is_flush: return (8 << 20) | (straight_hi << 16)
    if 4 in cnt:
        quad = max(r for r,c in rc.items() if c==4)
        kick = max(r for r,c in rc.items() if c!=4)
        return (7 << 20) | (quad<<16) | (kick<<12)
    if 3 in cnt and 2 in cnt:
        trip = max(r for r,c in rc.items() if c==3)
        pair = max(r for r,c in rc.items() if c==2)
        return (6 << 20) | (trip<<16) | (pair<<12)
    if is_flush:
        s = 5 << 20
        for i,r in enumerate(ranks[:5]): s |= r << (16-4*i)
        return s
    if straight_hi >= 0: return (4 << 20) | (straight_hi << 16)
    if 3 in cnt:
        trip = max(r for r,c in rc.items() if c==3)
        ks = sorted([r for r in ranks if r!=trip], reverse=True)
        return (3<<20) | (trip<<16) | (ks[0]<<12) | (ks[1]<<8)
    if cnt.count(2) >= 2:
        prs = sorted([r for r,c in rc.items() if c==2], reverse=True)
        k = max(r for r in ranks if r not in prs[:2])
        return (2<<20) | (prs[0]<<16) | (prs[1]<<12) | (k<<8)
    if 2 in cnt:
        pair = max(r for r,c in rc.items() if c==2)
        ks = sorted([r for r in ranks if r!=pair], reverse=True)
        return (1<<20) | (pair<<16) | (ks[0]<<12) | (ks[1]<<8) | (ks[2]<<4)
    s = 0
    for i,r in enumerate(ranks[:5]): s |= r << (16-4*i)
    return s


def fast_eval_7(cards_7):
    all_ranks = [c % 13 for c in cards_7]
    all_suits = [c // 13 for c in cards_7]
    best = 0
    for i in range(7):
        for j in range(i+1, 7):
            r5 = [all_ranks[k] for k in range(7) if k != i and k != j]
            s5 = [all_suits[k] for k in range(7) if k != i and k != j]
            r5.sort(reverse=True)
            s = _score_5(r5, s5)
            if s > best: best = s
    return best


def monte_carlo(hole1, hole2, n=500):
    used = set(hole1) | set(hole2)
    deck = [c for c in range(52) if c not in used]
    wins = ties = 0
    for _ in range(n):
        random.shuffle(deck)
        board = deck[:5]
        s1 = fast_eval_7(list(hole1) + board)
        s2 = fast_eval_7(list(hole2) + board)
        if s1 > s2: wins += 1
        elif s1 == s2: ties += 1
    return (wins + ties/2) / n


# ===== Worker for parallel computation =====
ALL_COMBOS = [type_combos(t) for t in range(N_TYPES)]

def compute_pair(args):
    """Worker function: compute equity for a single (i,j) pair"""
    i, j, n_iter = args
    ci = ALL_COMBOS[i]
    cj = ALL_COMBOS[j]
    c1 = ci[0]
    c2 = None
    for candidate in cj:
        if len(set(c1) & set(candidate)) == 0:
            c2 = candidate
            break
    if c2 is None:
        return (i, j, 0.5)
    avg = monte_carlo(c1, c2, n_iter)
    return (i, j, avg)


def precompute(n_iter=500):
    t0 = time.time()

    # Check for existing cache
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            eq = json.load(f)
        # Check if complete: scan all rows, not just last
        incomplete_rows = sum(1 for row in eq if sum(1 for v in row if v < 0.001) > 50)
        if incomplete_rows == 0:
            print(f"Loading complete cache: {CACHE}", flush=True)
            return eq
        print(f"Cache incomplete ({incomplete_rows}/169 rows), regenerating...", flush=True)

    # Generate all pairs to compute
    pairs = []
    for i in range(N_TYPES):
        for j in range(i, N_TYPES):
            pairs.append((i, j, n_iter))

    print(f"Precomputing {N_TYPES}x{N_TYPES} equity ({n_iter} MC/pair) "
          f"with {cpu_count()} CPUs, {len(pairs)} pairs...", flush=True)

    eq = [[0.0]*N_TYPES for _ in range(N_TYPES)]

    # Parallel computation
    completed = 0
    with Pool() as pool:
        # Process in batches for progress reporting
        batch_size = max(1, len(pairs) // 50)
        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start:batch_start + batch_size]
            results = pool.map(compute_pair, batch, chunksize=max(1, len(batch)//cpu_count()))
            for i, j, avg in results:
                eq[i][j] = avg
                eq[j][i] = 1.0 - avg
                completed += 1

            elapsed = time.time() - t0
            pct = completed / len(pairs) * 100
            if pct > 0:
                eta = elapsed / pct * (100 - pct)
                print(f"  {pct:.0f}% ({completed}/{len(pairs)}) elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)

            # Save checkpoint
            with open(CACHE, 'w') as f:
                json.dump(eq, f)

    with open(CACHE, 'w') as f:
        json.dump(eq, f)
    elapsed = time.time() - t0
    print(f"Done! {elapsed:.0f}s -> {CACHE}", flush=True)
    return eq


if __name__ == "__main__":
    eq = precompute(n_iter=500)
    cc = [type_combo_count(t) for t in range(N_TYPES)]
    total = sum(cc)
    ss = [sum(eq[t][j]*cc[j] for j in range(N_TYPES))/total for t in range(N_TYPES)]
    print(f"AA={ss[0]:.3f} AKs(24)={ss[24]:.3f} 72o(154)={ss[154]:.3f} AKo(102)={ss[102]:.3f}", flush=True)
