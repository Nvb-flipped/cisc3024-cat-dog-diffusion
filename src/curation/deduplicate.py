"""Near-duplicate families over exact official 32x32 training crops (pure pixel math).

Two crops join one family when any of these holds, also against the other's mirror image:
  * identical pixels;
  * pixel L2 distance < 4.0 on [0,1] RGB (the official evaluator's own near-duplicate radius,
    so the training set never contains copies the memorization test would merge);
  * 64-bit difference hash (9x8 grayscale) Hamming distance <= DHASH_RADIUS *and* pixel L2
    below CONFIRM_RADIUS, which catches reposts of one photo at different resolutions or
    recompressions. A hash match alone is not enough: plain studio backgrounds made a dog and a
    cat collide (L2 12.3) in the personally deduplicated v010 set.
No evaluator network (classifier or Inception) is involved. Families are suggestions for the
keep rule; distinct animals that merely look alike can share a family only through these pixel
criteria, which are strict.
"""
import numpy as np
import torch

L2_RADIUS = 4.0
DHASH_RADIUS = 4
CONFIRM_RADIUS = 8.0


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def dhash(pixels):
    """pixels: uint8 [N,32,32,3] -> uint64 [N] difference hashes (9x8 area-resized luma)."""
    gray = pixels.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    small = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(gray)[:, None], (8, 9))[:, 0].numpy()
    bits = (small[:, :, 1:] > small[:, :, :-1]).reshape(len(pixels), 64)
    return np.packbits(bits, axis=1, bitorder='little').view('<u8')[:, 0]


def popcount(values):
    values = values.astype(np.uint64)
    count = np.zeros(values.shape, dtype=np.int64)
    for shift in range(64):
        count += ((values >> np.uint64(shift)) & np.uint64(1)).astype(np.int64)
    return count


def families(pixels, chunk=2048, device='cpu'):
    """Return a family root index for every crop. pixels: uint8 numpy [N,32,32,3]."""
    n = len(pixels)
    uf = UnionFind(n)
    flat = torch.from_numpy(pixels).to(device).float().div(255).reshape(n, -1)
    mirrored = torch.from_numpy(np.ascontiguousarray(pixels[:, :, ::-1])).to(device).float().div(255).reshape(n, -1)
    hashes, mirror_hashes = dhash(pixels), dhash(np.ascontiguousarray(pixels[:, :, ::-1]))
    for start in range(0, n, chunk):
        block = flat[start:start + chunk]
        for reference in (flat, mirrored):
            distance = torch.cdist(block, reference)
            rows, cols = torch.nonzero(distance < L2_RADIUS, as_tuple=True)
            for i, j in zip((rows + start).tolist(), cols.tolist()):
                if i != j:
                    uf.union(i, j)
        for reference, pixels_ref in ((hashes, flat), (mirror_hashes, mirrored)):
            close = popcount(hashes[start:start + chunk, None] ^ reference[None, :]) <= DHASH_RADIUS
            for i, j in zip(*np.nonzero(close)):
                i, j = int(i + start), int(j)
                if i != j and torch.dist(flat[i], pixels_ref[j]) < CONFIRM_RADIUS:
                    uf.union(i, j)
    return [uf.find(i) for i in range(n)]
