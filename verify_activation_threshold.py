"""Independent finite enumeration for the residual Hall activation formula.

Thresholds are computed from cut inequalities, then compared with the first
positive support difference obtained by the companion integer-table enumerator.
Finite checks supplement the manuscript proof; they do not establish priority.
"""
from itertools import product
from math import ceil, comb, inf
from random import Random
from verify_support_perturbations import counts, finite_difference


def activation_threshold(k, h, q):
    n = len(k)
    edges = [(j, i) for i in range(n) for j in range(n) if i != j]
    allowed = [edge for bit, edge in enumerate(edges) if h & (1 << bit)]
    required = [edge for bit, edge in enumerate(edges) if q & (1 << bit)]
    incoming = [sum(i == v for j, i in required) for v in range(n)]
    outgoing = [sum(j == v for j, i in required) for v in range(n)]
    lower = max([1] + [ceil(v / d) for v, d in zip(incoming, k)]
                + [ceil(v / d) for v, d in zip(outgoing, k)])
    for mask in range(1 << n):
        receivers = {i for i in range(n) if mask & (1 << i)}
        neighbors = {j for j, i in allowed if i in receivers}
        slack = sum(k[j] for j in neighbors) - sum(k[i] for i in receivers)
        spill = sum(j in neighbors and i not in receivers for j, i in required)
        assert slack >= 0
        if slack == 0 and spill:
            return inf
        if slack:
            lower = max(lower, (spill + slack - 1) // slack)
    return lower


def mask_from_edges(n, selected):
    edges = [(j, i) for i in range(n) for j in range(n) if i != j]
    return sum(1 << edges.index(edge) for edge in selected)


def main():
    rng = Random(20261010)
    scale_checks = threshold_checks = 0
    for k in [(1, 1), (1, 2), (1, 1, 1), (1, 1, 2),
              (1, 2, 2), (2, 2, 2), (1, 1, 1, 1)]:
        n = len(k)
        bits = n * (n - 1)
        _, degrees, essential = counts(k)
        pairs = []
        if n <= 3:
            for states in product(range(3), repeat=bits):
                h = sum(1 << bit for bit, state in enumerate(states) if state)
                q = sum(1 << bit for bit, state in enumerate(states) if state == 2)
                if q and degrees[h]:
                    pairs.append((h, q))
        else:
            for _ in range(120):
                h = rng.randrange(1, 1 << bits)
                q = h & rng.randrange(1, 1 << bits)
                if q and degrees[h]:
                    pairs.append((h, q))
        max_scale = 6 if n <= 3 else 4
        dilated = {m: counts(tuple(m * v for v in k))[1]
                   for m in range(1, max_scale + 1)}
        for h, q in pairs:
            threshold = activation_threshold(k, h, q)
            assert (threshold == inf) == bool(q & ~essential[h])
            if threshold < inf:
                assert threshold <= q.bit_count()
                assert threshold <= max_scale
            for m, dm in dilated.items():
                diff = finite_difference(dm, h ^ q, q)
                assert (diff > 0) == (m >= threshold), (k, h, q, m, threshold)
                scale_checks += 1
            threshold_checks += 1

    # Ineffective dependencies at boundary margins (one-based labels in paper).
    n = 3
    e = mask_from_edges(n, [(0, 2), (1, 2), (2, 0), (2, 1)])
    h = (1 << 6) - 1
    for m in range(1, 9):
        _, dm, em = counts((m, m, 2 * m))
        assert dm[e] == dm[h] == comb(2 * m, m)
        assert em[h] == e

    # Two individually ineffective arcs with a collective exponential effect.
    n = 4
    e = mask_from_edges(n, [(0, 1), (1, 2), (2, 3), (3, 0)])
    q1 = mask_from_edges(n, [(0, 3)])
    q2 = mask_from_edges(n, [(2, 1)])
    h = e | q1 | q2
    assert activation_threshold((1,) * n, h, q1 | q2) == 1
    for m in range(1, 5):
        _, dm, em = counts((m,) * n)
        assert dm[e] == dm[e | q1] == dm[e | q2] == 1
        assert dm[h] == comb(2 * m, m)
        assert em[h] == h
        assert finite_difference(dm, e, q1 | q2) == dm[h] - 1

    assert activation_threshold((1, 1, 1), (1 << 6) - 1, (1 << 6) - 1) == 2
    print(f'Exact activation/inessentiality thresholds: {threshold_checks}')
    print(f'Threshold versus enumerated support differences: {scale_checks}')
    print('Ineffective, collective, and delayed-activation sparse examples: passed')


if __name__ == '__main__':
    main()
