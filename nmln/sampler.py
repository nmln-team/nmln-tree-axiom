import numpy as np

try:
    from numba import jit, njit
except Exception:
    # Fallback: decorator that does nothing if numba is unavailable/broken
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    njit = jit


#####################################################
# Contraction-Deletion Sampler 
#####################################################

def _indegree_matrix(w):
    return np.diag(np.sum(w, axis=1))


def _laplacian_reduced(w, r=0):
    L = _indegree_matrix(w) - w
    mask = np.arange(L.shape[0]) != r
    return L[np.ix_(mask, mask)]


def _delete_edge(w, e):
    i, j = e
    w_new = w.copy()
    w_new[i, j] = 0.0
    w_new[j, i] = 0.0
    return w_new


def _contract_edge(adj, e):
    """
    Contract edge e=(0, j), merging node j into node 0.
    New node k maps to old node k if k < j, old node k+1 if k >= j (for k >= 1).
    """
    _, j = e
    n = adj.shape[0]

    # Merged weights: combine connections of nodes 0 and j
    merged = adj[0].copy() + adj[j].copy()
    merged[0] = 0.0
    merged[j] = 0.0

    # Old indices for new nodes 1..n-2 (everything except 0 and j)
    keep = [k for k in range(1, n) if k != j]

    new_n = n - 1
    M = np.zeros((new_n, new_n))

    # Row/column 0: merged node
    for idx, old_k in enumerate(keep):
        M[0, idx + 1] = merged[old_k]
        M[idx + 1, 0] = merged[old_k]

    # Interior block: non-merged nodes
    for i1, ok1 in enumerate(keep):
        for i2, ok2 in enumerate(keep):
            M[i1 + 1, i2 + 1] = adj[ok1, ok2]

    return M


def _old_from_new(k, j):
    """Map node index in contracted graph back to original graph."""
    if k == 0:
        return 0
    return k if k < j else k + 1


def _sorted_edge(u, v):
    return (min(u, v), max(u, v))


def _build_contraction_map(w_before, e, prev_cm, rng):
    """
    Build contraction mapping: edges in contracted graph -> edges in original graph.
    For edges incident to the merged node, probabilistically disambiguate
    between the two original endpoints based on their weights.
    """
    _, j = e
    n_new = w_before.shape[0] - 1
    eps = 1e-15
    cm = {}

    for u in range(n_new):
        for v in range(u + 1, n_new):
            if u == 0:
                # Edge from merged node: disambiguate between node 0 and node j
                old_v = _old_from_new(v, j)
                w0 = w_before[0, old_v]
                wj = w_before[j, old_v]
                if rng.uniform() < w0 / (w0 + wj + eps):
                    mapped = _sorted_edge(0, old_v)
                else:
                    mapped = _sorted_edge(j, old_v)
            else:
                mapped = _sorted_edge(_old_from_new(u, j), _old_from_new(v, j))
            # Chain through previous CM to get original-graph edge
            cm[(u, v)] = prev_cm.get(mapped, mapped)

    return cm


def sampler(w, r=0, rng=None):
    """
    Sample a random spanning tree via contraction-deletion.
    Uses the Matrix Tree Theorem for edge inclusion probabilities.
    O(n^4) complexity. For better performance, use wilson_method_fast.
    """
    if rng is None:
        rng = np.random
    w = w.astype(np.float64)
    n = w.shape[0]
    edges = []
    cm = {(u, v): (u, v) for u in range(n) for v in range(u + 1, n)}

    i = 1
    while len(edges) < n - 1:
        # If only one edge left from root, it must be in the tree — include it
        if i >= w.shape[0] - 1:
            e = (0, w.shape[0] - 1)
            w_before = w
            w = _contract_edge(w, e)
            new_cm = _build_contraction_map(w_before, e, cm, rng)
            edges.append(cm.get(e, e))
            cm = new_cm
            i = 1
            continue

        e = (0, i)

        # Use slogdet for numerical stability (avoids overflow/underflow in det)
        sign_wo, logdet_wo = np.linalg.slogdet(_laplacian_reduced(_delete_edge(w, e), r=r))
        sign_full, logdet_full = np.linalg.slogdet(_laplacian_reduced(w, r=r))

        if sign_full <= 0:
            # Degenerate graph (disconnected or single node left), force include
            p_e = 1.0
        elif sign_wo <= 0:
            # No spanning trees without this edge — it must be included
            p_e = 1.0
        else:
            # P(e in T) = 1 - exp(logdet_without - logdet_full)
            log_ratio = logdet_wo - logdet_full
            p_e = np.clip(1.0 - np.exp(log_ratio), 0.0, 1.0)

        if rng.uniform() <= p_e:
            # Include edge: contract
            w_before = w
            w = _contract_edge(w, e)
            new_cm = _build_contraction_map(w_before, e, cm, rng)
            edges.append(cm.get(e, e))
            cm = new_cm
            i = 1
        else:
            # Exclude edge: delete
            w = _delete_edge(w, e)
            i += 1

    return edges

#####################################################
# Wilson's Algorithm - njit-compiled, exact (no uniform degradation)
#####################################################

@njit
def _seed_numba_rng(seed):
    """Seed numba's internal RNG.

    numba keeps a random state that is completely separate from NumPy's, and
    calling np.random.seed() from interpreted Python does NOT touch it. Seeding
    it requires a call from inside nopython mode, which is what this is for.
    """
    np.random.seed(seed)


@njit
def _wilson_sample(weight, r, soft_steps, hard_steps):
    """
    Numba-compiled Wilson's algorithm core.

    Every step is weighted. Earlier versions degraded to UNIFORM neighbour
    selection once a walk exceeded a step cap, which silently broke the
    weighted spanning-tree guarantee: the loop-erased walk was no longer the
    weighted random walk, so Wilson's theorem no longer applied and the
    returned tree came from an unspecified distribution. On sparse weight
    matrices it could even return edges of weight zero.

    Instead the walk now runs to completion -- Wilson's algorithm terminates
    with probability one on a connected graph with positive weights. Two
    diagnostics are returned rather than a silent repair:
      slow_walks : walks that needed more than `soft_steps` (for monitoring)
      status     : 0 ok, 1 exceeded `hard_steps`, 2 reached a node with no
                   positive-weight neighbour (the graph is disconnected)
    A non-zero status is raised by the caller; it is never papered over.

    Returns (edges, slow_walks, total_walks, status).
    """
    n = weight.shape[0]
    in_tree = np.zeros(n, dtype=np.bool_)
    in_tree[r] = True

    pos = np.full(n, -1, dtype=np.int64)
    path = np.empty(n, dtype=np.int64)
    edges = np.empty((n - 1, 2), dtype=np.int64)
    edge_count = 0
    slow_walks = 0
    total_walks = 0
    status = 0

    cumw = np.empty(n, dtype=np.float64)

    for start in range(n):
        if in_tree[start]:
            continue

        path_len = 0
        v = start
        pos[v] = 0
        path[0] = v
        path_len = 1
        steps = 0
        hit_slow = False
        total_walks += 1

        while not in_tree[v]:
            if steps >= hard_steps:
                status = 1
                break
            if steps >= soft_steps and not hit_slow:
                hit_slow = True
                slow_walks += 1

            # weighted sampling -- always, with no uniform degradation
            s = 0.0
            for j in range(n):
                if j != v:
                    s += weight[v, j]

            if s <= 0.0:
                status = 2
                break

            cumw[0] = weight[v, 0] if 0 != v else 0.0
            for j in range(1, n):
                cumw[j] = cumw[j - 1] + (weight[v, j] if j != v else 0.0)
            r_val = np.random.random() * s
            u = 0
            for j in range(n):
                if cumw[j] > r_val:
                    u = j
                    break
            else:
                u = n - 1

            v = u
            steps += 1

            if pos[v] != -1:
                cut = pos[v] + 1
                for idx in range(cut, path_len):
                    pos[path[idx]] = -1
                path_len = cut
            else:
                pos[v] = path_len
                path[path_len] = v
                path_len += 1

        if status != 0:
            break

        # add edges along path
        for idx in range(path_len - 1):
            a = path[idx]
            b = path[idx + 1]
            if a < b:
                edges[edge_count, 0] = a
                edges[edge_count, 1] = b
            else:
                edges[edge_count, 0] = b
                edges[edge_count, 1] = a
            edge_count += 1
            in_tree[a] = True
        in_tree[path[path_len - 1]] = True

        # reset positions
        for idx in range(path_len):
            pos[path[idx]] = -1

    return edges[:edge_count], slow_walks, total_walks, status


def wilson_method_fast(weight, r=0, seed=None, max_steps_factor=None, hard_limit_multiple=100_000,
                       raise_on_failure=True):
    """
    Sample a random spanning tree using Wilson's algorithm.
    Thin wrapper around numba-compiled core.
    max_steps_factor: multiplier for n to set per-walk step cap (default 2*n).
    raise_on_failure: if False, return (None, slow_walks, total_walks) instead of
    raising when the weights are pathological, so the caller can fall back to the
    exact matrix-tree sampler rather than losing the run.
    Returns (edges_list, slow_walks, total_walks); edges_list is None on failure.
    """
    if seed is not None:
        # Must go through a jitted call: the tree walk below runs in nopython
        # mode and draws from numba's RNG, not NumPy's.
        _seed_numba_rng(seed)
    weight = np.ascontiguousarray(weight, dtype=np.float64)
    n = weight.shape[0]
    # `soft` only marks a walk as slow, for monitoring. `hard` is a safety net
    # that raises: Wilson terminates with probability one, so exceeding it means
    # the weights are pathological, and that should be visible rather than
    # silently repaired into a draw from the wrong distribution.
    #
    # The multiple is large on purpose. Stalls are rare but real: a synthetic data
    # run survived ~789k tree draws before one walk needed more than 1000x the
    # soft cap, at a weight ratio of only 5e5. Expected hitting time grows with
    # that ratio, so a tight cap turns a once-per-million slow draw into a dead
    # job. At n=7 this allows ~9.8e6 steps, costing well under a second for an
    # event that rare; the error still fires if the weights become truly
    # degenerate, and matrix_tree remains the bounded-time exact alternative.
    # Cheap O(n^2) reachability check first: on a disconnected weighted graph no
    # spanning tree exists, and the walk would otherwise wander its own component
    # until the step cap and report a misleading "skewed weights" error.
    reach = np.zeros(n, dtype=np.bool_)
    reach[r] = True
    frontier = [r]
    while frontier:
        u = frontier.pop()
        for v in range(n):
            if not reach[v] and weight[u, v] > 0.0:
                reach[v] = True
                frontier.append(v)
    if not reach.all():
        if not raise_on_failure:
            return None, 0, 0
        raise RuntimeError(
            f"The weighted graph on n={n} nodes is disconnected under these "
            f"weights ({int((~reach).sum())} node(s) unreachable from root {r}), "
            f"so it has no spanning tree. Check for underflow in exp(potential)."
        )

    soft = 2 * n * n if max_steps_factor is None else int(n * max_steps_factor)
    hard = soft * hard_limit_multiple
    edges_arr, slow_walks, total_walks, status = _wilson_sample(weight, r, soft, hard)
    if status == 1:
        if not raise_on_failure:
            return None, int(slow_walks), int(total_walks)
        raise RuntimeError(
            f"Wilson's walk exceeded {hard} steps on an n={n} graph. The weights "
            f"are extremely skewed (max/min ratio "
            f"{weight.max() / weight[weight > 0].min():.3g}). Raise "
            f"hard_limit_multiple, or use the matrix-tree sampler, which is exact "
            f"in bounded O(n^4) time."
        )
    if status == 2:
        if not raise_on_failure:
            return None, int(slow_walks), int(total_walks)
        raise RuntimeError(
            f"Wilson's walk reached a node with no positive-weight neighbour on an "
            f"n={n} graph: the weighted graph is disconnected, so no spanning tree "
            f"exists under these weights. Check for underflow in exp(potential)."
        )
    edges_list = [(int(edges_arr[i, 0]), int(edges_arr[i, 1])) for i in range(edges_arr.shape[0])]
    return edges_list, int(slow_walks), int(total_walks)
