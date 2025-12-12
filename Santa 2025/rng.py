from config import math

class FastRNG:
    __slots__ = ("s0", "s1")
    
    def __init__(self, seed:int = 42):
        self.s0 = seed ^ 0x853c49e6748fea9b
        self.s1 = (seed * 0x9e3779b97f4a7c15) ^ 0xc4ceb9fe1a85ec53

        self.s0 &= (1 << 64) - 1
        self.s1 &= (1 << 64) - 1

    @staticmethod
    def _rotl(x: int, k: int) -> int:
        return ((x << k) & ((1 << 64) - 1)) | (x >> (64 - k))
    
    def next(self) -> int:
        s0 = self.s0
        s1 = self.s1

        r = (s0 + s1) & ((1 << 64) - 1)

        s1 ^= s0
        self.s0 = (self._rotl(s0, 24) ^ s1 ^ ((s1 << 16) & ((1 << 64) - 1))) & ((1 << 64) - 1)
        self.s1 = self._rotl(s1, 37) & ((1 << 64) - 1)

        return r
    
    def rf(self) -> float:
        return (self.next() >> 11) * (1.0 / (1 << 53))
    
    def ri(self, n: int) -> int:
        return int(self.next() % n)
    
    def gaussian(self) -> float:
        u1 = self.rf() + 1e-12
        u2 = self.rf()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def rf2(self) -> float:
        return self.rf() * 2.0 - 1.0