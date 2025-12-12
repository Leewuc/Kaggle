from config import List, dataclass, field, BBOX_EPS
from geometry import Poly, get_poly, overlap

@dataclass
class TreeState:
    x: float
    y: float
    a: float

@dataclass
class Cfg:
    n: int
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    a: List[float] = field(default_factory=list)
    pl: List[Poly] = field(default_factory=list)
    gx0: float = 0.0
    gy0: float = 0.0
    gx1: float = 0.0
    gy1: float = 0.0

    def __post_init__(self):
        if len(self.x) < self.n:
            self.x = (self.x + [0.0] * self.n)[:self.n]
        if len(self.y) < self.n:
            self.y = (self.y + [0.0] * self.n)[:self.n]
        if len(self.a) < self.n:
            self.a = (self.a + [0.0] * self.n)[:self.n]
        if not self.pl or len(self.pl) != self.n:
            self.pl = [get_poly(self.x[i], self.y[i], self.a[i]) for i in range(self.n)]
        
        self.update_global()
    
    def copy(self) -> "Cfg":
        new_x = self.x.copy()
        new_y = self.y.copy()
        new_a = self.a.copy()
        new_cfg = Cfg(n=self.n, x=new_x, y=new_y, a=new_a)
        # new_cfg.update_all()
        return new_cfg
    
    def update_poly(self, i: int) -> None:
        self.pl[i] = get_poly(self.x[i], self.y[i], self.a[i])
    
    def update_all(self) -> None:
        for i in range(self.n):
            self.update_poly(i)
        self.update_global()
    
    def update_global(self) -> None:
        if self.n == 0:
            self.gx0 = self.gy0 = 0.0
            self.gx1 = self.gy1 = 0.0
            return
        
        gx0 = gy0 = 1e9
        gx1 = gy1 = -1e9
        for i in range(self.n):
            p = self.pl[i]
            if p.x0 < gx0:
                gx0 = p.x0
            if p.x1 > gx1:
                gx1 = p.x1
            if p.y0 < gy0:
                gy0 = p.y0
            if p.y1 > gy1:
                gy1 = p.y1
        
        self.gx0, self.gy0, self.gx1, self.gy1 = gx0, gy0, gx1, gy1
    
    def has_overlap(self, i: int) -> bool:
        pi = self.pl[i]
        for j in range(self.n):
            if i == j:
                continue
            if overlap(pi, self.pl[j]):
                return True
        return False
    
    def any_overlap(self) -> bool:
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if overlap(self.pl[i], self.pl[j]):
                    return True
        return False
    
    def side(self) -> float:
        return max(self.gx1 - self.gx0, self.gy1 - self.gy0)
    
    def score(self) -> float:
        if self.n <= 0:
            return 0.0
        s = self.side()
        return s * s / self.n
    
    def get_boundary_indices(self) -> List[int]:
        res: List[int] = []
        eps = BBOX_EPS
        for i in range(self.n):
            p = self.pl[i]
            if (p.x0 - self.gx0 < eps or self.gx1 - p.x1 < eps or
                p.y0 - self.gy0 < eps or self.gy1 - p.y1 < eps):
                res.append(i)
        return res
    
    def clone_without_tree(self, remove_idx: int) -> "Cfg":
        xs: List[float] = []
        ys: List[float] = []
        as_: List[float] = []
        for i in range(self.n):
            if i == remove_idx:
                continue
            xs.append(self.x[i])
            ys.append(self.y[i])
            as_.append(self.a[i])
        new_cfg = Cfg(n=self.n - 1, x=xs, y=ys, a=as_)
        new_cfg.update_all()
        return new_cfg
        