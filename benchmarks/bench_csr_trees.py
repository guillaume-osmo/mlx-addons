"""CSR/segment-scatter ExtraTrees vs sklearn on Apple silicon.

Reproduces the numbers in the commit message. sklearn is pinned to one thread so the
comparison is per-core-fair; with n_jobs=-1 it narrows but the shape holds.
"""
import time

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from mlx_addons.ensemble import ExtraTreesRegressorMLXCSR

CASES = [(3_000, 10, 100), (20_000, 20, 100), (50_000, 20, 100),
         (100_000, 30, 100), (200_000, 30, 100), (100_000, 30, 300)]

rng = np.random.default_rng(0)
print(f"{'n':>8} {'p':>4} {'T':>5} {'MLX s':>8} {'skl s':>8} {'ratio':>8} {'MLX R2':>9} {'skl R2':>9}")
for n, p, T in CASES:
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = (X[:, :5] @ rng.normal(size=5)).astype(np.float32)
    t = time.time()
    m = ExtraTreesRegressorMLXCSR(n_estimators=T, max_depth=10, random_state=0).fit(X, y)
    tm = time.time() - t
    r2m = 1 - np.var(y - np.asarray(m.predict(X)).ravel()) / np.var(y)
    t = time.time()
    s = ExtraTreesRegressor(n_estimators=T, max_depth=10, random_state=0, n_jobs=1).fit(X, y)
    ts = time.time() - t
    print(f"{n:8d} {p:4d} {T:5d} {tm:8.2f} {ts:8.2f} {tm/ts:7.2f}x {r2m:+9.4f} {s.score(X, y):+9.4f}")
