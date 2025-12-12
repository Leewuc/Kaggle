from config import math, EPS
from state import Cfg

def squeeze(cfg: Cfg) -> Cfg:
    c = cfg.copy()
    cx = (c.gx0 + c.gx1) * 0.5
    cy = (c.gy0 + c.gy1) * 0.5

    scale = 0.9995
    while scale >= 0.98:
        trial = c.copy()
        for i in range(trial.n):
            trial.x[i] = cx + (c.x[i] - cx) * scale
            trial.y[i] = cy + (c.y[i] - cy) * scale
        trial.update_all()

        if not trial.any_overlap():
            c = trial
        else:
            break
        scale -= 0.0005
    return c

def compaction(cfg: Cfg, iters: int) -> Cfg:
    c = cfg.copy()
    best_side = c.side()
    steps = [0.02, 0.008, 0.003, 0.001, 0.0004]

    for _ in range(iters):
        cx = (c.gx0 + c.gx1) * 0.5
        cy = (c.gy0 + c.gy1) * 0.5
        improved = False

        for i in range(c.n):
            ox = c.x[i]
            oy = c.y[i]
            dx = cx - c.x[i]
            dy = cy - c.y[i]
            d = math.hypot(dx,dy)
            if d < 1e-6:
                continue

            for step in steps:
                c.x[i] = ox + dx / d * step
                c.y[i] = oy + dy / d * step
                c.update_poly(i)

                if not c.has_overlap(i):
                    c.update_global()
                    s = c.side()
                    if s < best_side - 1e-12:
                        best_side = s
                        improved = True
                        ox = c.x[i]
                        oy = c.y[i]
                    else:
                        c.x[i] = ox
                        c.y[i] = oy
                        c.update_poly(i)
                        c.update_global()
                else:
                    c.x[i] = ox
                    c.y[i] = oy
                    c.update_poly(i)
        c.update_global()
        if not improved:
            break
    return c

def local_search(cfg: Cfg, max_iter: int) -> Cfg:
    c = cfg.copy()
    best_side = c.side()

    steps = [0.01, 0.004, 0.002, 0.001, 0.0005, 0.00025, 0.0001]
    rots = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125]

    dxs = [1, -1, 0, 0, 1, 1, -1, -1]
    dys = [0, 0, 1, -1, 1, -1, 1, -1]

    for _ in range(max_iter):
        improved = False

        for i in range(c.n):
            cx = (c.gx0 + c.gx1) * 0.5
            cy = (c.gy0 + c.gy1) * 0.5
            ddx = cx - c.x[i]
            ddy = cy - c.y[i]
            dist = math.hypot(ddx, ddy)

            if dist > 1e-6:
                for st in steps:
                    ox = c.x[i]
                    oy = c.y[i]

                    c.x[i] += ddx / dist * st
                    c.y[i] += ddy / dist * st
                    c.update_poly(i)

                    if not c.has_overlap(i):
                        c.update_global()
                        s = c.side()
                        if s < best_side - 1e-12:
                            best_side = s
                            improved = True
                        else:
                            c.x[i] = ox
                            c.y[i] = oy
                            c.update_poly(i)
                            c.update_global()
                    else:
                        c.x[i] = ox
                        c.y[i] = oy
                        c.update_poly(i)
            
            for st in steps:
                for d in range(8):
                    ox = c.x[i]
                    oy = c.y[i]

                    c.x[i] += dxs[d] * st
                    c.y[i] += dys[d] * st
                    c.update_poly(i)

                    if not c.has_overlap(i):
                        c.update_global()
                        s = c.side()
                        if s < best_side - 1e-12:
                            best_side = s
                            improved = True
                        else:
                            c.x[i] = ox
                            c.y[i] = oy
                            c.update_poly(i)
                            c.update_global()
                    else:
                        c.x[i] = ox
                        c.y[i] = oy
                        c.update_poly(i)
            
            for rt in rots:
                for da in (rt, -rt):
                    oa = c.a[i]
                    c.a[i] = (c.a[i] + da) % 360.0
                    c.update_poly(i)

                    if not c.has_overlap(i):
                        c.update_global()
                        s = c.side()
                        if s < best_side - 1e-12:
                            best_side = s
                            improved = True
                        else:
                            c.a[i] = oa
                            c.update_poly(i)
                            c.update_global()
                    else:
                        c.a[i] = oa
                        c.update_poly(i)
        if not improved:
            break
    return c

def edge_slide_compaction(cfg: Cfg, outer_iter: int) -> Cfg:
    c = cfg.copy()
    best_side = c.side()

    for _ in range(outer_iter):
        improved = False

        for i in range(c.n):
            gcx = (c.gx0 + c.gx1) * 0.5
            gcy = (c.gy0 + c.gy1) * 0.5
            dirs = [
                (gcx - c.x[i], gcy - c.y[i]),
                (1.0, 0.0),
                (-1.0, 0.0),
                (0.0, 1.0),
                (0.0, -1.0)
            ]

            for dx, dy in dirs:
                length = math.hypot(dx, dy)
                if length < 1e-9:
                    continue
                dx /= length
                dy /= length

                max_step = 0.30
                lo = 0.0
                hi = max_step
                best_step = 0.0

                ox = c.x[i]
                oy = c.y[i]

                for _ in range(20):
                    mid = 0.5 * (lo + hi)
                    c.x[i] = ox + dx * mid
                    c.y[i] = oy + dy * mid
                    c.update_poly(i)
                    c.update_global()
                    ok_overlap = not c.has_overlap(i)
                    ok_side = (c.side() <= best_side + 1e-9)

                    if ok_overlap and ok_side:
                        best_step = mid
                        lo = mid
                    else:
                        hi = mid
                
                if best_step > 1e-6:
                    c.x[i] = ox + dx * best_step
                    c.y[i] = oy + dy * best_step
                    c.update_poly(i)
                    c.update_global()
                    
                    ns = c.side()
                    if ns < best_side - 1e-12:
                        best_side = ns
                        improved = True
                else:
                    c.x[i] = ox
                    c.y[i] = oy
                    c.update_poly(i)
                    c.update_global()
        if not improved:
            break
    return c