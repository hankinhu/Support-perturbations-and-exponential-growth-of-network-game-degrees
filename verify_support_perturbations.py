"""Exact finite-case checks for the support-perturbation theorem.

Uses only the Python standard library. Enumerates balanced tables independently
of coefficient extraction and checks support-difference identities and activation.
These finite checks supplement, but do not replace, the proofs in the manuscript.
"""

from functools import lru_cache
from itertools import product
from math import comb, factorial, pi, sqrt
from random import Random


def bounded_compositions(total, bounds):
    if not bounds:
        if total == 0:
            yield ()
        return
    for first in range(min(total, bounds[0]) + 1):
        for rest in bounded_compositions(total - first, bounds[1:]):
            yield (first,) + rest


@lru_cache(None)
def counts(k):
    n = len(k)
    edge_index = {(j, i): t for t, (i, j) in enumerate(
        (i, j) for i in range(n) for j in range(n) if i != j
    )}
    size = 1 << (n * (n - 1))
    exact = [0] * size

    def visit(i, columns, support, weight):
        if i == n:
            if not any(columns):
                exact[support] += weight
            return
        neighbors = [j for j in range(n) if j != i]
        for row in bounded_compositions(k[i], tuple(columns[j] for j in neighbors)):
            remaining = list(columns)
            next_support = support
            denominator = 1
            for j, amount in zip(neighbors, row):
                remaining[j] -= amount
                denominator *= factorial(amount)
                if amount:
                    next_support |= 1 << edge_index[j, i]
            visit(i + 1, tuple(remaining), next_support,
                  weight * (factorial(k[i]) // denominator))

    visit(0, k, 0, 1)
    degrees = exact.copy()
    essential = [mask if value else 0 for mask, value in enumerate(exact)]
    for bit in range(n * (n - 1)):
        for mask in range(size):
            if mask & (1 << bit):
                degrees[mask] += degrees[mask ^ (1 << bit)]
                essential[mask] |= essential[mask ^ (1 << bit)]
    return exact, degrees, essential


def subsets(mask):
    part = mask
    while True:
        yield part
        if part == 0:
            break
        part = (part - 1) & mask


def finite_difference(degrees, e, q):
    return sum((-1) ** (q.bit_count() - r.bit_count()) * degrees[e | r]
               for r in subsets(q))


def direct_required_sum(exact, h, q):
    return sum(exact[s] for s in subsets(h) if s & q == q)


def main():
    rng = Random(20261008)
    identity_checks = equality_checks = activation_checks = 0
    for k in ((1, 1), (1, 2), (1, 1, 1), (1, 1, 2),
              (1, 2, 2), (2, 2, 2), (1, 1, 1, 1), (1, 2, 1, 2)):
        n = len(k)
        bits = n * (n - 1)
        _, base_degrees, base_essential = counts(k)
        if n <= 3:
            pairs = []
            for states in product(range(3), repeat=bits):
                e = sum(1 << bit for bit, state in enumerate(states) if state == 1)
                q = sum(1 << bit for bit, state in enumerate(states) if state == 2)
                if q and base_degrees[e | q]:
                    pairs.append((e, q))
        else:
            pairs = []
            for _ in range(100):
                h = rng.randrange(1, 1 << bits)
                q = h & rng.randrange(1, 1 << bits)
                if q and base_degrees[h]:
                    pairs.append((h ^ q, q))

        for m in (1, 2, 3):
            exact, degrees, essential = counts(tuple(m * value for value in k))
            assert essential == base_essential, (k, m, 'essential-support invariance')
            for e, q in pairs:
                h = e | q
                diff = finite_difference(degrees, e, q)
                assert diff == direct_required_sum(exact, h, q), (k, m, e, q)
                identity_checks += 1
                if q & ~essential[h]:
                    assert diff == 0, (k, m, e, q, 'forced zero')
                elif m >= q.bit_count():
                    assert diff > 0, (k, m, e, q, 'activation')
                    activation_checks += 1
                if base_degrees[e]:
                    same = base_degrees[e] == base_degrees[h]
                    assert same == (base_essential[e] == base_essential[h])
                    assert same == (degrees[e] == degrees[h])
                    equality_checks += 1

        # Independently construct a witness at the claimed activation bound.
        # Summing an integral base-margin table for each required edge proves
        # feasibility at |Q| without enumerating unnecessarily large dilations.
        base_exact, _, _ = counts(k)
        for e, q in pairs:
            h = e | q
            if q & ~base_essential[h]:
                continue
            for bit in range(bits):
                if q & (1 << bit):
                    assert any(base_exact[s] and s & (1 << bit) for s in subsets(h))

    # Complete loopless three-player graph: first joint activation is m=2.
    full = (1 << 6) - 1
    for m in range(1, 11):
        exact, degrees, _ = counts((m, m, m))
        franel = sum(comb(m, t) ** 3 for t in range(m + 1))
        assert degrees[full] == franel
        assert exact[full] == franel - 2
    ratios = []
    for m in (20, 50, 100):
        franel = sum(comb(m, t) ** 3 for t in range(m + 1))
        prediction = 2 * 8 ** m / (pi * sqrt(3) * m)
        ratios.append((m, franel / prediction))
    print(f'Exact support-difference identities: {identity_checks}')
    print(f'Equality/essential-support/dilation equivalences: {equality_checks}')
    print(f'Finite activation checks at tested dilations: {activation_checks}')
    print('Three-player first activation and boundary deficit: passed (m=1,...,10)')
    print(f'Growth-ratio prefactor checks (exact/predicted): {ratios}')


if __name__ == '__main__':
    main()
