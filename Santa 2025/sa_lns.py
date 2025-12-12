from config import (
    math, List, EPS,
    SA_T0, SA_TM, SA_ITER_BASE, SA_RESTART_BASE,
    REMOVE_RATIO, REINSERT_TRY, VERBOSE, MAX_N
)
from rng import FastRNG
from state import Cfg, TreeState
from moves import squeeze, compaction, local_search, edge_slide_compaction

def sa_opt(cfg: Cfg, iters: int, T0: float, Tm: float, seed: int) -> Cfg:
    rng = FastRNG(seed)
    best = cfg.copy()
    cur = cfg.copy()

    best_side = best.side()
    cur_side = best_side

    T = T0
    alpha = (Tm / T0) ** (1.0 / max(1, iters))
    no_imp = 0

    for it in range(iters):
        mt = rng.ri(10)
        scale = T /T0
        valid = True

        if mt == 0:
            i = rng.ri(cur.n)
            ox, oy = cur.x[i], cur.y[i]
            cur.x[i] += rng.gaussian() * 0.5 * scale
            cur.y[i] += rng.gaussian() * 0.5 * scale
            cur.update_poly(i)
            if cur.has_overlap(i):
                cur.x[i], cur.y[i] = ox, oy
                cur.update_poly(i)
                valid = False
        
        elif mt == 1:
            i = rng.ri(cur.n)
            ox, oy = cur.x[i], cur.y[i]
            bcx = (cur.gx0 + cur.gx1) * 0.5
            bcy = (cur.gy0 + cur.gy1) * 0.5
            dx = bcx - cur.x[i]
            dy = bcy - cur.y[i]
            d = math.hypot(dx, dy)
            if d > 1e-6:
                cur.x[i] += dx / d * rng.rf() * 0.6 * scale
                cur.y[i] += dy / d * rng.rf() * 0.6 * scale
            cur.update_poly(i)
            if cur.has_overlap(i):
                cur.x[i], cur.y[i] = ox, oy
                cur.update_poly(i)
                valid = False
        
        elif mt == 2:
            i = rng.ri(cur.n)
            oa = cur.a[i]
            cur.a[i] = (cur.a[i] + rng.gaussian() * 80.0 * scale) % 360.0
            cur.update_poly(i)
            if cur.has_overlap(i):
                cur.a[i] = oa
                cur.update_poly(i)
                valid = False
        
        elif mt == 3:
            i = rng.ri(cur.n)
            ox, oy, oa = cur.x[i], cur.y[i], cur.a[i]
            cur.x[i] += rng.rf2() * 0.5 * scale
            cur.y[i] += rng.rf2() * 0.5 * scale
            cur.a[i] = (cur.a[i] + rng.rf2() * 60.0 * scale) % 360.0
            cur.update_poly(i)
            if cur.has_overlap(i):
                cur.x[i], cur.y[i], cur.a[i] = ox, oy, oa
                cur.update_poly(i)
                valid = False
        
        elif mt == 4:
            boundary = cur.get_boundary_indices()
            if not boundary:
                valid = False
            else:
                i = boundary[rng.ri(len(boundary))]
                ox, oy, oa = cur.x[i], cur.y[i], cur.a[i]
                bcx = (cur.gx0 + cur.gx1) * 0.5
                bcy = (cur.gy0 + cur.gy1) * 0.5
                dx = bcx - cur.x[i]
                dy = bcy - cur.y[i]
                d = math.hypot(dx, dy)
                if d > 1e-6:
                    cur.x[i] += dx / d * rng.rf() * 0.7 * scale
                    cur.y[i] += dy / d * rng.rf() * 0.7 * scale
                cur.a[i] = (cur.a[i] + rng.rf2() * 50.0 * scale) % 360.0
                cur.update_poly(i)
                if cur.has_overlap(i):
                    cur.x[i], cur.y[i], cur.a[i] = ox, oy, oa
                    cur.update_poly(i)
                    valid = False
        
        elif mt == 5:
            factor = 1.0 - rng.rf() * 0.004 * scale
            cx = (cur.gx0 + cur.gx1) * 0.5
            cy = (cur.gy0 + cur.gy1) * 0.5
            trial = cur.copy()
            for i in range(trial.n):
                trial.x[i] = cx + (trial.x[i] - cx) * factor
                trial.y[i] = cy + (trial.y[i] - cy) * factor
            trial.update_all()
            if not trial.any_overlap():
                cur = trial
            else:
                valid = False
        
        elif mt == 6:
            i = rng.ri(cur.n)
            ox, oy = cur.x[i], cur.y[i]
            levy = (rng.rf() + 0.001) ** -1.3 * 0.008
            cur.x[i] += rng.rf2() * levy
            cur.y[i] += rng.rf2() * levy
            cur.update_poly(i)
            if cur.has_overlap(i):
                cur.x[i], cur.y[i] = ox, oy
                cur.update_poly(i)
                valid = False
        
        elif mt == 7 and cur.n > 1:
            i = rng.ri(cur.n)
            j = (i + 1) % cur.n
            oxi, oyi = cur.x[i], cur.y[i]
            oxj, oyj = cur.x[j], cur.y[j]
            dx = rng.rf2() * 0.3 * scale
            dy = rng.rf2() * 0.3 * scale
            cur.x[i] += dx
            cur.y[i] += dy
            cur.x[j] += dx
            cur.y[j] += dy
            cur.update_poly(i)
            cur.update_poly(j)
            if cur.has_overlap(i) or cur.has_overlap(j):
                cur.x[i], cur.y[i] = oxi, oyi
                cur.x[j], cur.y[j] = oxj, oyj
                cur.update_poly(i)
                cur.update_poly(j)
                valid = False

        else:
            i = rng.ri(cur.n)
            ox, oy = cur.x[i], cur.y[i]
            cur.x[i] += rng.rf2() * 0.002
            cur.y[i] += rng.rf2() * 0.002
            cur.update_poly(i)
            if cur.has_overlap(i):
                cur.x[i], cur.y[i] = ox, oy
                cur.update_poly(i)
                valid = False
        
        if not valid:
            no_imp += 1
            T *= alpha
            if T < Tm:
                T = Tm
            continue

        cur.update_global()
        ns = cur.side()
        delta = ns - cur_side

        if delta < 0 or rng.rf() < math.exp(-delta / max(T, 1e-12)):
            cur_side = ns
            if ns < best_side:
                best_side = ns
                best = cur.copy()
                no_imp = 0
            else:
                no_imp += 1
        else:
            cur = best.copy()
            cur_side = best_side
            no_imp += 1
        
        if no_imp > 200:
            T = min(T * 5.0, T0)
            no_imp = 0
        T *= alpha
        if T < Tm:
            T = Tm
    return best

def perturb(cfg: Cfg, strength: float, rng: FastRNG) -> Cfg:
    c = cfg.copy()
    original = cfg.copy()

    np = max(1, int(c.n * 0.08 + strength * 3))
    for _ in range(np):
        i = rng.ri(c.n)
        c.x[i] += rng.gaussian() * strength * 0.5
        c.y[i] += rng.gaussian() * strength * 0.5
        c.a[i] = (c.a[i] + rng.gaussian() * 30.0) % 360.0
    c.update_all()

    for _ in range(150):
        fixed = True
        for i in range(c.n):
            if c.has_overlap(i):
                fixed = False
                cx = (c.gx0 + c.gx1) * 0.5
                cy = (c.gy0 + c.gy1) * 0.5
                dx = c.x[i] - cx
                dy = c.y[i] - cy
                d = math.hypot(dx, dy)
                if d > 1e-6:
                    c.x[i] += dx / d * 0.02
                    c.y[i] += dy / d * 0.02
                c.a[i] = (c.a[i] + rng.rf2() * 15.0) % 360.0
                c.update_poly(i)
        if fixed:
            break
    
    c.update_global()
    if c.any_overlap():
        return original
    return c

def optimize_parallel(cfg: Cfg, iters: int, restarts: int) -> Cfg:
    global_best = cfg.copy()
    global_best_side = global_best.side()
    rng_global = FastRNG(42 + cfg.n)

    for r in range(restarts):
        if r == 0:
            start = cfg.copy()
        else:
            start = perturb(cfg, 0.02 + 0.02 * (r % 8), rng_global)
            if start.any_overlap():
                continue
        
        seed = 42 + r * 1000 + cfg.n
        o = sa_opt(start, iters, SA_T0, SA_TM, seed)
        o = squeeze(o)
        o = compaction(o, 50)
        o = edge_slide_compaction(o, 10)
        o = local_search(o, 80)

        if not o.any_overlap():
            s = o.side()
            if s < global_best_side:
                global_best_side = s
                global_best = o.copy()
                if VERBOSE:
                    print(f"[optimize_parallel] restart {r}: side = {s:.6f}")
    
    global_best = squeeze(global_best)
    global_best = compaction(global_best, 80)
    global_best = edge_slide_compaction(global_best, 12)
    global_best = local_search(global_best, 150)

    if global_best.any_overlap():
        return cfg
    return global_best

def compute_free_area(cfg: Cfg) -> List[float]:
    n = cfg.n
    free_area = [0.0] * n
    area = [0.0] * n
    overlap_sum = [0.0] * n

    for i in range(n):
        p = cfg.pl[i]
        w = max(0.0, p.x1 - p.x0)
        h = max(0.0, p.y1 - p.y0)
        area[i] = w * h
    
    for i in range(n):
        pi = cfg.pl[i]
        for j in range(n):
            if i == j:
                continue
            pj = cfg.pl[j]
            ix0 = max(pi.x0, pj.x0)
            iy0 = max(pi.y0, pj.y0)
            ix1 = min(pi.x1, pj.x1)
            iy1 = min(pi.y1, pj.y1)
            dx = ix1 - ix0
            dy = iy1 - iy0
            if dx > 0.0 and dy > 0.0:
                overlap_sum[i] += dx * dy
    
    for i in range(n):
        occ = min(overlap_sum[i], area[i])
        free_area[i] = max(0.0, area[i] - occ)
    return free_area    

def compute_protrude_score(cfg: Cfg) -> List[float]:
    n = cfg.n
    score = [0.0] * n
    cx = (cfg.gx0 + cfg.gx1) * 0.5
    cy = (cfg.gy0 + cfg.gy1) * 0.5
    side = cfg.side()
    eps = side * 0.02

    for i in range(n):
        p = cfg.pl[i]
        on_boundary = (
            p.x0 - cfg.gx0 < eps
            or cfg.gx1 - p.x1 < eps
            or p.y0 - cfg.gy0 < eps
            or cfg.gy1 - p.y1 < eps
        )
        if not on_boundary:
            score[i] = 0.0
            continue

        tx = 0.5 * (p.x0 + p.x1)
        ty = 0.5 * (p.y0 + p.y1)
        d = math.hypot(tx - cx, ty - cy)
        score[i] = d
    return score

def reinsert_trees(base: Cfg, removed: List[TreeState], seed: int) -> Cfg:
    cur = base.copy()
    rng = FastRNG(seed)

    for t in removed:
        if cur.n >= MAX_N:
            return base
        
        cur.x.append(t.x)
        cur.y.append(t.y)
        cur.a.append(t.a)
        cur.n += 1
        cur.pl.append(cur.pl[0])
        idx = cur.n - 1
        cur.update_poly(idx)
        cur.update_global()

        placed = False
        for _ in range(REINSERT_TRY):
            if not cur.has_overlap(idx):
                placed = True
                break

            cx = (cur.gx0 + cur.gx1) * 0.5
            cy = (cur.gy0 + cur.gy1) * 0.5
            radius = 0.1 + 0.6 * rng.rf()
            ang = 2.0 * math.pi * rng.rf()

            cur.x[idx] = cx + radius * math.cos(ang)
            cur.y[idx] = cy + radius * math.sin(ang)
            cur.a[idx] = (t.a + rng.rf2() * 120.0) % 360.0

            cur.update_poly(idx)
            cur.update_global()
        
        if not placed:
            cur.x.pop()
            cur.y.pop()
            cur.a.pop()
            cur.pl.pop()
            cur.n -= 1
            return base
    if cur.any_overlap():
        return base
    return cur

def free_area_heuristic(cfg: Cfg, remove_ratio: float = REMOVE_RATIO, seed: int = 1234567) -> Cfg:
    best = cfg.copy()
    n = cfg.n
    if n <= 5:
        return best
    
    k = int(math.floor(n * remove_ratio + 1e-9))
    if k < 1:
        k = 1
    if k >= n:
        k = n - 1
    
    free_area = compute_free_area(cfg)
    protrude_score = compute_protrude_score(cfg)

    free_list = sorted(
        [(free_area[i], i) for i in range(n)],
        key=lambda x: (-x[0], x[1])
    )

    prot_list = sorted(
        [(protrude_score[i], i) for i in range(n) if protrude_score[i] > 0.0],
        key=lambda x: (-x[0], x[1])
    )

    k_prot = min(len(prot_list), (k * 2) // 3)
    k_free = k - k_prot
    if k_free < 0:
        k_free = 0
    
    remove_flag = [False] * n
    removed: List[TreeState] = []

    removed_cnt = 0
    for sc, idx in prot_list:
        if removed_cnt >= k_prot:
            break
        if remove_flag[idx]:
            continue
        remove_flag[idx] = True
        removed.append(TreeState(cfg.x[idx],cfg.y[idx],cfg.a[idx]))
        removed_cnt += 1
    
    for sc, idx in free_list:
        if removed_cnt >= k:
            break
        if remove_flag[idx]:
            continue
        remove_flag[idx] = True
        removed.append(TreeState(cfg.x[idx],cfg.y[idx],cfg.a[idx]))
        removed_cnt += 1
    
    if not removed:
        return best
    
    xs, ys, as_ = [], [], []
    for i in range(n):
        if not remove_flag[i]:
            xs.append(cfg.x[i])
            ys.append(cfg.y[i])
            as_.append(cfg.a[i])
    
    reduced = Cfg(n=n - len(removed), x=xs, y=ys, a=as_)
    reduced.update_all()
    if reduced.any_overlap():
        return best
    
    iters = max(2000,8000)
    reduced_opt = optimize_parallel(reduced, iters, 8)
    with_inserted = reinsert_trees(reduced_opt, removed, seed + n * 101)
    if with_inserted.n != n or with_inserted.any_overlap():
        return best
    
    with_inserted = squeeze(with_inserted)
    with_inserted = compaction(with_inserted, 40)
    with_inserted = edge_slide_compaction(with_inserted, 10)
    with_inserted = local_search(with_inserted, 80)

    if (not with_inserted.any_overlap() and 
        with_inserted.side() < best.side() - 1e-12):
        return with_inserted
    return best

