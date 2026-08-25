"""ExtraTrees on MLX with a CSR/segment formulation — no index matrix, zero padding.

The two earlier designs both carry a dense `(m, L)` index matrix padded to the largest
node at each level. Measured padding waste: 12-13x serial, 31-41x once the forest is
batched (because L becomes the forest-wide max child size). That dense gather, not the
host syncs, was the real cost — removing 990 of 1000 syncs made it *slower*.

This version keeps no index matrix at all. It holds one node-assignment vector and
updates it elementwise per level:

    node <- 2*node + 1 + (x[row, feat[node]] >= thr[node])

Every per-node statistic is then a segment reduction over that vector, and MLX has all
three natively as single scatters:

    sum  ->  mx.zeros(n).at[gid].add(v)        (this is exactly mlx_graphs.scatter_add)
    min  ->  mx.full(n, +inf).at[gid].minimum(v)
    max  ->  mx.full(n, -inf).at[gid].maximum(v)

Note mlx_graphs.scatter_max instead builds an (out_size, n_edges) mask; ArrayAt.maximum
does it in one pass, so it is used directly here.

Memory per level is O(T*N), independent of depth and of node-size skew.
Padding waste: 0x.
"""

from __future__ import annotations

from typing import Optional, Union

import mlx.core as mx
import numpy as np

INF = np.float32(3.0e38)
NEG_INF = -1.0e30


def _resolve_k(max_features: Union[int, float, str, None], d: int) -> int:
    if max_features is None or max_features in ("all", 1.0):
        return d
    if max_features == "sqrt":
        return max(1, int(np.sqrt(d)))
    if max_features == "log2":
        return max(1, int(np.log2(d)))
    if isinstance(max_features, float):
        return max(1, int(round(max_features * d)))
    return max(1, min(int(max_features), d))


class ExtraTreesRegressorMLXCSR:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        max_features: Union[int, float, str, None] = 1.0,
        min_samples_split: int = 2,
        var_eps: float = 0.0,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.max_features = max_features
        self.min_samples_split = int(min_samples_split)
        self.var_eps = float(var_eps)
        self.random_state = int(random_state)
        self.feature_: Optional[mx.array] = None
        self.threshold_: Optional[mx.array] = None
        self.value_: Optional[mx.array] = None
        self.n_syncs_ = 0

    def fit(self, X, y) -> "ExtraTreesRegressorMLXCSR":
        Xn = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        yn = np.ascontiguousarray(np.asarray(y, dtype=np.float32)).ravel()
        N, D = Xn.shape
        T, MD = self.n_estimators, self.max_depth
        K = _resolve_k(self.max_features, D)
        max_nodes = (1 << (MD + 1)) - 1
        NT = T * max_nodes

        Xm, ym = mx.array(Xn), mx.array(yn)
        Xb = mx.broadcast_to(Xm.reshape(1, N, D), (T, N, D))
        y2 = mx.broadcast_to(ym.reshape(1, N), (T, N))
        yf = y2.reshape(-1)
        tree_base = (mx.arange(T, dtype=mx.int32) * max_nodes).reshape(T, 1)

        feature = mx.full((T, max_nodes), -1, dtype=mx.int32)
        threshold = mx.zeros((T, max_nodes), dtype=mx.float32)
        value = mx.zeros((T, max_nodes), dtype=mx.float32)

        node = mx.zeros((T, N), dtype=mx.int32)      # node id within its tree
        key = mx.random.key(self.random_state)
        self.n_syncs_ = 0

        zf = mx.zeros((NT,), dtype=mx.float32)

        for depth in range(MD + 1):
            gid = (tree_base + node).reshape(-1)                       # (T*N,)
            cnt = zf.at[gid].add(mx.ones_like(yf))
            sy = zf.at[gid].add(yf)
            sy2 = zf.at[gid].add(yf * yf)
            cs = mx.maximum(cnt, 1.0)
            mean = sy / cs
            var = mx.maximum(sy2 / cs - mean * mean, 0.0)

            # write the mean for every node; the traversal only reads it at leaves
            value = value + (mean.reshape(T, max_nodes) - value) * (cnt.reshape(T, max_nodes) > 0)

            if depth == MD:
                break

            can = (cnt >= float(self.min_samples_split)) & (var > self.var_eps)

            best_sc = mx.full((NT,), NEG_INF, dtype=mx.float32)
            best_f = mx.zeros((NT,), dtype=mx.int32)
            best_t = mx.zeros((NT,), dtype=mx.float32)

            for _ in range(K):
                key, k1, k2 = mx.random.split(key, 3)
                # one candidate feature per (tree, node), broadcast down to rows
                fpn = mx.random.randint(0, D, shape=(NT,), key=k1).astype(mx.int32)
                fr = mx.take(fpn, gid).reshape(T, N, 1)
                x = mx.take_along_axis(Xb, fr, axis=2).reshape(-1)     # (T*N,)

                lo = mx.full((NT,), INF, dtype=mx.float32).at[gid].minimum(x)
                hi = mx.full((NT,), -INF, dtype=mx.float32).at[gid].maximum(x)
                u = mx.random.uniform(shape=(NT,), key=k2)
                thr = lo + (hi - lo) * u

                m = (x < mx.take(thr, gid)).astype(mx.float32)
                nl = zf.at[gid].add(m)
                sl = zf.at[gid].add(m * yf)
                nr = cnt - nl
                sr = sy - sl
                sse = sl * sl / mx.maximum(nl, 1.0) + sr * sr / mx.maximum(nr, 1.0)
                ok = (nl > 0.0) & (nr > 0.0) & (hi > lo)
                sc = mx.where(ok, sse, mx.array(NEG_INF, dtype=mx.float32))

                take = sc > best_sc
                best_sc = mx.where(take, sc, best_sc)
                best_f = mx.where(take, fpn, best_f)
                best_t = mx.where(take, thr, best_t)

            split = can & (best_sc > (NEG_INF / 2.0))
            feature = mx.where(split.reshape(T, max_nodes), best_f.reshape(T, max_nodes), feature)
            threshold = mx.where(split.reshape(T, max_nodes), best_t.reshape(T, max_nodes), threshold)

            # route every row in one elementwise step — this replaces the whole
            # partition/compaction machinery of the padded versions
            fr = mx.take(best_f, gid).reshape(T, N, 1)
            xr = mx.take_along_axis(Xb, fr, axis=2).reshape(T, N)
            tr = mx.take(best_t, gid).reshape(T, N)
            sp = mx.take(split.astype(mx.int32), gid).reshape(T, N) > 0
            child = 2 * node + 1 + (xr >= tr).astype(mx.int32)
            node = mx.where(sp, mx.minimum(child, max_nodes - 1), node)

        mx.eval(feature, threshold, value)
        self.feature_, self.threshold_, self.value_ = feature, threshold, value
        return self

    def predict(self, X) -> np.ndarray:
        Xm = mx.array(np.asarray(X, dtype=np.float32))
        N, D = map(int, Xm.shape)
        T, max_nodes = map(int, self.feature_.shape)
        Xb = mx.broadcast_to(Xm.reshape(1, N, D), (T, N, D))
        node = mx.zeros((T, N), dtype=mx.int32)
        for _ in range(self.max_depth + 1):
            feat = mx.take_along_axis(self.feature_, node, axis=1).astype(mx.int32)
            thr = mx.take_along_axis(self.threshold_, node, axis=1)
            leaf = feat < 0
            xf = mx.take_along_axis(Xb, mx.maximum(feat, 0).reshape(T, N, 1), axis=2).reshape(T, N)
            ch = mx.minimum(2 * node + 1 + (xf >= thr).astype(mx.int32), max_nodes - 1)
            node = mx.where(leaf, node, ch)
        pred = mx.mean(mx.take_along_axis(self.value_, node, axis=1), axis=0)
        mx.eval(pred)
        return np.asarray(pred)
