from config import (
    csv, Dict, time, sys,
    SA_ITER_BASE, SA_RESTART_BASE, 
    BACKPROP_MAX_PASS, BACKPROP_RANGE,
    VERBOSE
)

from state import Cfg
from sa_lns import optimize_parallel, free_area_heuristic
from moves import squeeze, compaction, edge_slide_compaction, local_search
from rng import FastRNG

def load_csv(path: str) -> Dict[int, Cfg]:
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        tmp = {}
        for row in reader:
            id_ = row["id"]
            x_s = row["x"].lstrip("s")
            y_s = row["y"].lstrip("s")
            d_s = row["deg"].lstrip("s")

            n = int(id_[:3])
            idx = int(id_.split("_")[1])

            if n not in tmp:
                tmp[n] = []
            tmp[n].append((idx, float(x_s), float(y_s), float(d_s)))
        
        for n, arr in tmp.items():
            xs = [0.0] * n
            ys = [0.0] * n
            as_ = [0.0] * n
            for idx, x, y, d in arr:
                if 0 <= idx < n:
                    xs[idx] = x
                    ys[idx] = y
                    as_[idx] = d
            cfg = Cfg(n=n, x=xs, y=ys, a=as_)
            cfg.update_all()
            data[n] = cfg
    return data

def save_csv(path: str, cfgs: Dict[int, Cfg]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "deg"])
        for n in range(1, 201):
            if n not in cfgs:
                continue
            c = cfgs[n]
            for i in range(n):
                writer.writerow([
                    f"{n:03d}_{i}",
                    f"s{c.x[i]:.15f}",
                    f"s{c.y[i]:.15f}",
                    f"s{c.a[i]:.15f}"
                ])

def phase1_optimize(cfgs: Dict[int, Cfg], base_iters: int, base_restarts: int) -> Dict[int, Cfg]:
    res: Dict[int, Cfg] = {}
    total_improved = 0

    print("Phase 1: Parallel-like optimization per n (Python single-process)")

    for n in range(200, 0, -1):
        if n not in cfgs:
            continue
        c = cfgs[n]
        oscore = c.score()

        iters = base_iters
        restarts = base_restarts

        if n <= 10:
            iters = int(iters * 2.5)
            restarts = int(restarts * 2)
        elif n <= 30:
            iters = int(iters * 1.8)
            restarts = int(restarts * 1.5)
        elif n <= 60:
            iters = int(iters * 1.3)
        elif n > 150:
            iters = int(iters * 0.7)
            restarts = int(restarts * 0.8)
        
        restarts = max(4, restarts)

        if VERBOSE:
            print(f"[Phase1] n={n:03d}, iters={iters}, restarts={restarts}, oscore={oscore:.6f}")
        
        o = optimize_parallel(c, iters, restarts)

        for m, pc in res.items():
            if m > n and m <= n + 2:
                if pc.n < n:
                    continue
                xs = pc.x[:n]
                ys = pc.y[:n]
                as_ = pc.a[:n]
                ad = Cfg(n=n, x=xs, y=ys, a=as_)
                ad.update_all()
                if not ad.any_overlap():
                    ad = compaction(ad, 40)
                    ad = edge_slide_compaction(ad, 8)
                    ad = local_search(ad, 60)
                    if not ad.any_overlap() and ad.side() < o.side():
                        o = ad
        
        if o.any_overlap() or o.side() > c.side() + 1e-14:
            o = c
        
        if not o.any_overlap() and n >= 10:
            seed = 1234567 + n * 101
            oh = free_area_heuristic(o, remove_ratio=0.50, seed=seed)
            if not oh.any_overlap() and oh.side() < o.side() - 1e-12:
                o = oh
        
        res[n] = o
        nscore = o.score()

        if nscore < oscore - 1e-10:
            total_improved += 1
            print(f"  n={n:3d}: {oscore:.6f} -> {nscore:.6f} ({(oscore - nscore) / oscore * 100.0:.4f}%)")
    
    print(f"Phase 1 completed: improved {total_improved} configurations.")
    return res

def phase2_backprop(res: Dict[int, Cfg]) -> None:
    print("Phase 2: Aggressive back propagation (removing trees) ... \n")

    backprop_improved = 0
    changed = True
    pass_num = 0

    while changed and pass_num < BACKPROP_MAX_PASS:
        changed = False
        pass_num += 1
        print(f"Pass {pass_num} ...")

        for k in range(200, 1, -1):
            if k not in res or (k - 1) not in res:
                continue

            cfg_k = res[k]
            cfg_k1 = res[k-1]
            side_k = cfg_k.side()
            side_k1 = cfg_k1.side()

            if side_k >= side_k1 - 1e-12:
                continue
            best_side = side_k1
            best_cfg = cfg_k1

            for remove_idx in range(cfg_k.n):
                reduced = cfg_k.clone_without_tree(remove_idx)
                if reduced.any_overlap():
                    continue
                reduced = squeeze(reduced)
                reduced = compaction(reduced, 60)
                reduced = edge_slide_compaction(reduced, 10)
                reduced = local_search(reduced, 100)

                if not reduced.any_overlap():
                    s = reduced.side()
                    if s < best_side - 1e-12:
                        best_side = s
                        best_cfg = reduced
            
            if best_side < side_k1 - 1e-12:
                old_score = res[k - 1].score()
                new_score = best_cfg.score()
                res[k - 1] = best_cfg
                print(
                    f"  [k vs k-1] n={k-1:3d}: {old_score:.6f} -> {new_score:.6f} "
                    f"(from n={k}, {(old_score - new_score) / old_score * 100.0:.4f}%)"
                )
                backprop_improved += 1
                changed = True
        
        for k in range(200, 2, -1):
            if k not in res:
                continue

            for src in range(k + 1, min(200, k + BACKPROP_RANGE) + 1):
                if src not in res:
                    continue

                cfg_k = res[k]
                cfg_src = res[src]
                side_k = cfg_k.side()
                side_src = cfg_src.side()

                if side_src >= side_k - 1e-12:
                    continue 
                to_remove = src - k
                if to_remove <= 0:
                    continue

                best_side = side_k
                best_cfg = cfg_k

                combos = generate_combos_for_backprop(src, to_remove, k)
                rng = FastRNG(seed=k * 1000 + src)

                for combo in combos:
                    remove_set = set(combo)
                    xs, ys, as_ = [], [], []
                    for i in range(cfg_src.n):
                        if i in remove_set:
                            continue
                        xs.append(cfg_src.x[i])
                        ys.append(cfg_src.y[i])
                        as_.append(cfg_src.a[i])
                    if len(xs) != k:
                        continue   

                    reduced = Cfg(n=k, x=xs, y=ys, a=as_)
                    reduced.update_all()
                    if reduced.any_overlap():
                        continue

                    reduced = squeeze(reduced)
                    reduced = compaction(reduced, 50)
                    reduced = edge_slide_compaction(reduced, 10)
                    reduced = local_search(reduced, 80)

                    if not reduced.any_overlap():
                        s = reduced.side()
                        if s < best_side - 1e-12:
                            best_side = s
                            best_cfg = reduced
                
                if best_side < side_k - 1e-12:
                    old_score = res[k].score()
                    new_score = best_cfg.score()
                    res[k] = best_cfg
                    print(
                        f"  [k vs src>k] n={k:3d}: {old_score:.6f} -> {new_score:.6f} "
                        f"(from n={src}, {(old_score - new_score) / old_score * 100.0:.4f}%)"
                    )
                    backprop_improved += 1
                    changed = True
        print(f" Pass {pass_num} completed. changed={changed}\n")
    print(f"Phase 2 completed: improved {backprop_improved} configurations.\n")

def generate_combos_for_backprop(src: int, to_remove: int, k: int):
    combos = []
    if to_remove == 1:
        combos = [[i] for i in range(src)]
    
    elif to_remove == 2 and src <= 50:
        for i in range(src):
            for j in range(i + 1, src):
                combos.append([i, j])
    
    else:
        max_trials = min(200, src * 3)
        rng = FastRNG(seed=src * 10007 + k)
        seen = set()
        for _ in range(max_trials):
            idxs = set()
            while len(idxs) < to_remove:
                idxs.add(rng.ri(src))
            combo = sorted(idxs)
            key = tuple(combo)
            if key in seen:
                continue
            seen.add(key)
            combos.append(combo)
    return combos

def main():
    in_path = ""
    out_path = ""
    iters = SA_ITER_BASE
    restarts = SA_RESTART_BASE

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-i" and i + 1 < len(args):
            in_path = args[i + 1]
            i += 2
        elif a == "-o" and i + 1 < len(args):
            out_path = args[i + 1]
            i += 2
        elif a == "-n" and i + 1 < len(args):
            iters = int(args[i + 1])
            i += 2
        elif a == "-r" and i + 1 < len(args):
            restarts = int(args[i + 1])
            i += 2
        else:
            i += 1
    print(f"Tree Packer (Python) - SA + LNS + Back-prop")
    print(f"Input: {in_path}, Output: {out_path}")
    print(f"Base Iterations: {iters}, Base Restarts: {restarts}\n")

    cfgs = load_csv(in_path)
    if not cfgs:
        print("No data loaded!")
        return
    
    print(f"Loaded {len(cfgs)} configurations.\n")
    init_score = sum(c.score() for c in cfgs.values())
    print(f"Initial total score: {init_score:.6f}\n")
    t0 = time.time()
    res = phase1_optimize(cfgs, iters, restarts)
    phase2_backprop(res)
    final_score = sum(c.score() for c in res.values())
    elapsed = time.time() - t0

    print("========================================")
    print(f"Initial: {init_score:.6f}")
    print(f"Final:   {final_score:.6f}")
    print(f"Improve: {init_score - final_score:.6f} "
          f"({(init_score - final_score) / init_score * 100.0:.4f}%)")
    print(f"Time:    {elapsed:.1f}s")
    print("========================================")

    save_csv(out_path, res)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()