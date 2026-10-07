import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import random
import typing as t
import numpy as np
from nmln import ontology
from nmln.sampler import wilson_method_fast, sampler as matrix_tree_sampler


class NewOneOfNConstraint:
    def __init__(
        self,
        predicates_names: t.List[str],
        ontology: ontology.Ontology,
        symmetric: bool = False,
        antireflexive: bool = False,
        alsoNone: bool = False,
        seed: int = 0,
    ):
        # Own RNG stream: the global NumPy/Python streams are also drawn from
        # by libraries in the training stack, which breaks reproducibility.
        self._rng = np.random.default_rng(seed)
        self.symmetric = symmetric
        self.antireflexive = antireflexive
        self.alsoNone = alsoNone
        self.alternatives = len(predicates_names)
        self.ontology = ontology

        s = self.ontology._dict_indices[predicates_names[0]].shape

        for predicate in predicates_names:
            if not self.ontology._dict_indices[predicate].shape == s:
                raise ValueError(
                    "Mutual exclusivity is not allowed among these predicates. Non compatible arity."
                )

        arity = len(s)

        # Res is a list of lists, where each inner list contains the indices of atoms that are bound together
        # If an element is not in a set, it means that it is cannot be changed
        res = [[k] for k in range(self.ontology.num_constants**arity)]
        if arity == 2:

            if self.antireflexive:
                # Here we cycle all the blocks of res.
                # For each block, if we remove the blocks containing indices on the diagonal (=reflexive)
                def constrains_symmetric(block):
                    for idx in block:
                        if idx % (self.ontology.num_constants + 1) == 0:
                            return True
                    return False

                res = [block for block in res if not constrains_symmetric(block)]

            if self.symmetric:
                # Here we cycle all the blocks of res.
                # For each block, we create a new block containing the original block and all the symmetric of the original
                # We also remove the symmetric blocks from the original list
                # TODO: this is not generic
                new_res = []
                cache = set()
                for block in res:
                    new_block = []
                    for idx in block:
                        i = idx // self.ontology.num_constants
                        j = idx % self.ontology.num_constants
                        symmetric_idx = j * self.ontology.num_constants + i
                        if idx not in cache and symmetric_idx not in cache:
                            cache.add(idx)
                            cache.add(symmetric_idx)
                            new_block.extend([idx, symmetric_idx])
                        else:
                            break
                    if len(new_block) > 0:
                        new_res.append(new_block)
                res = new_res

        else:
            print(
                "Warning: Symmetric and antireflexive constraints are only supported for binary predicates"
            )

        # Make it a numpy array (Valid only for antireflexive + symmetric constraints)  TODO: only for blocks of size 2 now
        res = np.array(res)

        # Repeating the current indexing scheme for all the predicates (for mutual exclusivity among them)
        res = np.tile(np.expand_dims(res, -2), [1, len(predicates_names), 1])
        base = [
            [
                [np.reshape(self.ontology._dict_indices[predicate], [-1])[0]]
                for predicate in predicates_names
            ]
        ]
        integer_indices = res + base
        self.integer_indices = integer_indices

        # Translating indices in the boolean scheme (one_hot)
        binary_indices = np.eye(self.ontology.linear_size)[integer_indices]

        # Combining boolean values for symmetrical states (in all but arity=2 is a size=1 dimension)
        self.binary_masks = torch.from_numpy(np.sum(binary_indices, axis=-2))

    def generate_noise_coherent_with_transformations(self, data, p_noise):
        self.binary_masks = self.binary_masks.to(data.device)

        data = data.detach().clone()

        if p_noise > 0:
            for m in self.binary_masks:
                m2 = m.sum(0)

                for k in range(len(data)):
                    if self._rng.random() < p_noise:
                        t = data[k]
                        t = torch.where(m2 > 0, torch.zeros_like(t), t)
                        data[k] = t + m[self._rng.choice(len(m), size=1)][0]

        return data


class TreeConstraint:
    @staticmethod
    def _connected(adj):
        """True if every node is reachable from 0 over strictly positive weights."""
        n = adj.shape[0]
        # Fast path. Weights are clamped non-negative, so a complete off-diagonal
        # is a complete graph, which is connected -- and that is the case on
        # essentially every draw. It is worth spending one vectorised pass to
        # reach that conclusion: the Python BFS below measured two to three times
        # the cost of the Wilson draw it guards at n=12-20.
        if np.count_nonzero(adj) == n * n - n:
            return True
        reach = np.zeros(n, dtype=bool)
        reach[0] = True
        frontier = [0]
        while frontier:
            u = frontier.pop()
            nbrs = np.nonzero((adj[u] > 0.0) & ~reach)[0]
            for v in nbrs:
                reach[v] = True
                frontier.append(int(v))
        return bool(reach.all())

    def __init__(
        self,
        predicate_name: t.List[str],
        ontology: ontology.Ontology,
        sampler_type: str = "wilson",
        seed: int = 0,
        wilson_ratio_limit: float = 1e6,
    ):
        # Own RNG stream, and an explicit per-draw seed for the tree sampler:
        # Wilson's core is numba-jitted and draws from numba's RNG, which no
        # amount of process-level seeding reaches.
        self._rng = np.random.default_rng(seed)
        # Wilson's walk now always runs to completion, so these are pure
        # diagnostics: a slow walk is one that needed more than 2n^2 steps,
        # which is the regime where the weights are becoming badly skewed.
        self.tree_draws = 0
        self.slow_walks = 0
        self.total_walks = 0
        # Wilson's expected hitting time grows with the spread of the edge
        # weights, so on a badly conditioned matrix it is not merely slow, it is
        # unusable. Above this ratio fall back to the
        # matrix-tree sampler, which is exact in bounded O(n^4) time.
        #
        # The choice is made from the WEIGHTS ALONE, before any randomness is
        # drawn, which is what keeps it exact. Switching *after* observing a
        # stall would not be: conditioning on "the walk stalled" correlates with
        # the tree that walk was going to produce, biasing the mixture.
        self.wilson_ratio_limit = wilson_ratio_limit
        self.matrix_tree_fallbacks = 0
        self.underflow_repairs = 0
        self.symmetric = True
        self.antireflexive = True
        self.ontology = ontology
        self.predicate_name = predicate_name
        self.sampler_type = sampler_type

        s = self.ontology._dict_indices[self.predicate_name[0]].shape
        arity = len(s)

        assert arity == 2, "Tree constraints are only supported for binary predicates"

        # All the atom indices in a single binary predicate
        to_filter = np.reshape(
            np.arange(self.ontology.num_constants**arity),
            [self.ontology.num_constants for _ in range(arity)],
        )

        # Here we remove the indices in the diagonal of the matrix of indices
        to_filter[np.diag_indices_from(to_filter)] = -1

        to_filter = to_filter[to_filter >= 0]

        self.indices_tree = to_filter

    def generate_noise_coherent_with_transformations(self, data, p_noise):
        # self.binary_masks = self.binary_masks.to(data.device)
        #
        # if p_noise > 0:
        #     for m in self.binary_masks:
        #
        #         for k in range(len(data)):
        #             if np.random.rand() < p_noise:
        #                 adj_matrix = np.random.randint(0, 2, self.ontology._dict_indices[self.predicate_name].shape)
        #                 adj_list = matrix_to_adj_list(adj_matrix)
        #                 adj_matrix_tree = adj_to_matrix(adj_list)
        #                 data[k][self.integer_indices] = torch.tensor(adj_matrix_tree).long().flatten()
        return data

    @staticmethod
    def ensure_symmetry(w):
        # Ensure the matrix is symmetric and does not contain NaNs or infs
        w = torch.where(torch.isnan(w), torch.zeros_like(w), w)
        w = torch.where(torch.isinf(w), torch.zeros_like(w), w)
        w_new = (w + w.T) / 2
        return w_new

    def gibbs_sampling_step(self, samples, potential):
        """
        Gibbs step enforcing a spanning-tree constraint with labeled edges.

        Works for any number of binary edge-label predicates (l >= 1):
          - Step 1: build weight matrix W(i,j) = Σ_k exp(ψ(rho_k)) / exp(ψ(zeros))
            where rho_k = state with only k-th predicate active on all off-diagonal pairs,
            and zeros = state with all bond predicates inactive.
          - Sample a spanning tree from W using the configured sampler.
          - Step 2: for each tree edge (i,j), sample a label k proportional to
            exp(ψ(rho_k))[chain, edge_idx], then set predicate k active for that edge
            (both directions, i.e. symmetric storage).

        Generalises the previous l=2 hardcoded version to arbitrary l while remaining fully backward-compatible with the l=2 molecule case (double/single bonds).
        """
        original_samples_shape = samples.shape
        samples = samples.reshape(-1, samples.shape[-1])

        assert (
            len(potential.potentials) == 1
        ), "Tree constraints are only supported for one potential function"
        local_potential = potential.potentials[0]
        assert (
            local_potential.k == 2
        ), "Tree constraints are only supported for k=2 potentials"

        num_labels = len(self.predicate_name)
        num_constants = self.ontology.num_constants
        device = samples.device
        eye = torch.eye(num_constants, device=device)

        # ── zeros state: all bond predicates set to 0 ─────────────────────────
        zeros = torch.clone(samples)
        for k in range(num_labels):
            zeros[:, self.ontology._dict_indices[self.predicate_name[k]]] = 0.0

        # ── rho_k states: only k-th predicate active (all off-diagonal = 1) ──
        # Start each from `zeros` so only one predicate is active at a time.
        rhos = []
        for k in range(num_labels):
            rho_k = torch.clone(zeros)
            rho_k[:, self.ontology._dict_indices[self.predicate_name[k]]] = (
                1.0 - eye
            )
            rhos.append(rho_k)

        # ── Compute log ψ(rho_k) for every k, and log ψ(zeros) ──────────────
        # Work entirely in log-space to avoid exp() underflow/overflow when
        # potential values are large in magnitude.
        ground_zeros = local_potential.ground(zeros)
        log_zeros = local_potential.call_on_groundings(ground_zeros).squeeze(-1)  # [batch, E]

        # log ψ(rho_k) for each label k; shape [batch, E]
        log_rhos = [
            local_potential.call_on_groundings(
                local_potential.ground(rho_k)
            ).squeeze(-1)
            for rho_k in rhos
        ]

        # log W(i,j) = logsumexp_k(log ψ(rho_k)) - log ψ(zeros)  [batch, E]
        log_rhos_stacked = torch.stack(log_rhos, dim=0)          # [L, batch, E]
        log_numerator = torch.logsumexp(log_rhos_stacked, dim=0)  # [batch, E]
        log_W = log_numerator - log_zeros                          # [batch, E]

        # Normalise per chain by subtracting the max log-weight (spanning-tree
        # sampling is scale-invariant, so this is lossless).
        log_W_max = log_W.max(dim=-1, keepdim=True).values
        potential_on_groundings = torch.exp(log_W - log_W_max)    # [batch, E] in (0,1]
        potential_on_groundings = torch.nan_to_num(
            potential_on_groundings, nan=0.0, posinf=1.0, neginf=0.0
        )

        # ── Per-chain loop ─────────────────────────────────────────────────────
        for chain in range(potential_on_groundings.shape[0]):

            # Build the n×n weight matrix for this chain
            off_diag_mask = ~torch.eye(num_constants, dtype=torch.bool, device=device)
            vals = potential_on_groundings[chain]
            adjacency = torch.zeros(num_constants, num_constants, device=device)
            adjacency[off_diag_mask] = torch.where(
                torch.isfinite(vals), vals, torch.zeros_like(vals)
            )

            adjacency = TreeConstraint.ensure_symmetry(adjacency)
            adjacency = torch.clamp(adjacency, min=0.0)
            adjacency = torch.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)

            # Fallback: if all weights are zero use uniform (keeps diagonal = 0)
            if torch.all(adjacency == 0):
                adjacency = 1.0 - eye

            # ── Sample a spanning tree ────────────────────────────────────────
            adj_np = adjacency.cpu().detach().numpy()
            adj_np = np.nan_to_num(adj_np, nan=0.0, posinf=0.0, neginf=0.0)
            self.tree_draws += 1

            # Underflow repair. log_W is normalised so the largest weight is 1,
            # so any edge more than ~745 nats below the maximum exponentiates to
            # exactly 0. If that disconnects the graph there is no spanning tree
            # and both samplers are undefined -- and the max/min test below
            # cannot see it, because the offending entries are no longer
            # positive, so the worst-conditioned graphs are exactly the ones it
            # waves through. Lift the off-diagonal to a floor 1e-12 under the
            # maximum: any tree using a floored edge is then still 1e-12 lighter
            # than one that avoids it, which is far below the precision the
            # sampler works at, so the sampled distribution is unchanged and the
            # graph is connected by construction.
            if not TreeConstraint._connected(adj_np):
                adj_np = np.maximum(adj_np, adj_np.max() * 1e-12)
                np.fill_diagonal(adj_np, 0.0)
                self.underflow_repairs += 1

            use_matrix_tree = self.sampler_type == "matrix_tree"
            if not use_matrix_tree:
                positive = adj_np[adj_np > 0]
                if positive.size:
                    ratio = float(adj_np.max() / positive.min())
                    if ratio > self.wilson_ratio_limit:
                        use_matrix_tree = True
                        self.matrix_tree_fallbacks += 1
            if not use_matrix_tree:
                # A Wilson draw that fails is a pathology of the weights, not of
                # the model: matrix-tree samples the same distribution exactly in
                # bounded time, so fall back for this draw and keep the counter
                # rather than ending a multi-hour run.
                indices_one_tree, n_slow, n_walks = wilson_method_fast(
                    adj_np, r=0, seed=int(self._rng.integers(0, 2**31 - 1)),
                    raise_on_failure=False,
                )
                self.slow_walks += n_slow
                self.total_walks += n_walks
                if indices_one_tree is None:
                    use_matrix_tree = True
                    self.matrix_tree_fallbacks += 1
            if use_matrix_tree:
                indices_one_tree = matrix_tree_sampler(adj_np, r=0, rng=self._rng)

            # ── Step 2: label each tree edge ──────────────────────────────────
            # One adjacency matrix per label; all start at zero.
            label_adjs = [
                torch.zeros(num_constants, num_constants, device=device)
                for _ in range(num_labels)
            ]

            for edge in indices_one_tree:
                row, col = edge
                index_edge = TreeConstraint.edge_index(row, col, num_constants)
                if index_edge < 0 or index_edge >= log_rhos[0].shape[1]:
                    continue

                # Stable softmax over labels using log ψ(rho_k) values
                log_lp = [log_rhos[k][chain, index_edge].item() for k in range(num_labels)]
                log_lp = [v if np.isfinite(v) else -1e9 for v in log_lp]
                max_lp = max(log_lp)
                label_potentials = [np.exp(v - max_lp) for v in log_lp]

                total = sum(label_potentials)
                if total == 0.0:
                    probs = [1.0 / num_labels] * num_labels
                else:
                    probs = [p / total for p in label_potentials]

                # Categorical sample: chosen label gets the edge (symmetric)
                chosen_k = int(self._rng.choice(num_labels, p=probs))
                label_adjs[chosen_k][row, col] = 1.0
                label_adjs[chosen_k][col, row] = 1.0

            # ── Write label adjacencies back into samples[chain, ...] ─────────
            for k in range(num_labels):
                pred_indices = self.ontology._dict_indices[self.predicate_name[k]]
                samples[chain, pred_indices] = label_adjs[k].float()

        samples = samples.reshape(original_samples_shape)
        return samples

    @staticmethod
    def edge_index(row, col, size):
        if row == col:
            raise ValueError("Diagonal elements are excluded.")
        # Defensive: ensure index is in bounds
        idx = row * (size - 1) + col - (col > row)
        return idx




if __name__ == "__main__":

    from nmln import ontology

    people = ontology.Domain("people", ["a", "b", "c"])
    smokes = ontology.Predicate("smokes", [people])
    friendOf = ontology.Predicate("friendOf", [people, people])
    enemyOf = ontology.Predicate("enemyOf", [people, people])
    indifferentTo = ontology.Predicate("indifferentTo", [people, people])

    o = ontology.Ontology(
        domains=[people],
        predicates=[smokes, friendOf, enemyOf, indifferentTo],
    )

    c = OneOfNConstraint(
        predicates_names=["smokes", "friendOf", "enemyOf", "indifferentTo"],
        ontology=o,
        symmetric=True,
        antireflexive=True,
    )

    c1 = NewOneOfNConstraint(
        predicates_names=["smokes", "friendOf", "enemyOf", "indifferentTo"],
        ontology=o,
        symmetric=True,
        antireflexive=True,
    )

    assert np.all(c.integer_indices == c1.integer_indices)
