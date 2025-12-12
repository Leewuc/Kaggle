from config import math, NV, PI, List, dataclass

TX: List[float] = [
    0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075,
    -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125
]

TY: List[float] = [
    0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2,
    -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5
]

@dataclass
class Poly:
    px: List[float]
    py: List[float]
    x0: float
    y0: float
    x1: float
    y1: float

def get_poly(cx: float, cy: float, deg: float) -> Poly:
    rad = deg * (PI / 180.0)
    s = math.sin(rad)
    c = math.cos(rad)
    px: List[float] = []
    py: List[float] = []

    minx = 1e9
    miny = 1e9
    maxx = -1e9
    maxy = -1e9

    for i in range(NV):
        x = TX[i] * c - TY[i] * s + cx
        y = TX[i] * s + TY[i] * c + cy
        px.append(x)
        py.append(y)

        if x < minx:
            minx = x
        if x > maxx:
            maxx = x
        if y < miny:
            miny = y
        if y > maxy:
            maxy = y
    return Poly(px=px, py=py, x0=minx, y0=miny, x1=maxx, y1=maxy)

def pip(px:float, py: float, poly:Poly) -> bool:
    inside = False
    j = NV - 1
    for i in range(NV):
        yi = poly.py[i]
        yj = poly.py[j]
        xi = poly.px[i]
        xj = poly.px[j]

        intersect = ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-18) + xi)

        if intersect:
            inside = not inside
        j = i
    return inside

def seg_intersect(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float) -> bool:
    d1 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
    d2 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
    d3 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    d4 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)

    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)

def overlap(a: Poly, b: Poly) -> bool:
    if a.x1 < b.x0 or b.x1 < a.x0 or a.y1 < b.y0 or b.y1 < a.y0:
        return False
    
    for i in range(NV):
        if pip(a.px[i], a.py[i], b):
            return True
        if pip(b.px[i], b.py[i], a):
            return True
    
    for i in range(NV):
        ni = (i + 1) % NV
        ax, ay = a.px[i], a.py[i]
        bx, by = a.px[ni], a.py[ni]

        for j in range(NV):
            nj = (j + 1) % NV
            cx, cy = b.px[j], b.py[j]
            dx, dy = b.px[nj], b.py[nj]

            if seg_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                return True
    return False