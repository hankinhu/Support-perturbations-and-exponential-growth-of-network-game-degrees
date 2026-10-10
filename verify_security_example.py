"""Exact checks for the illustrative interdependent-security realization.

Standard library only. All four indifference systems are solved completely;
pure-profile averaging independently checks the mixed cost-gap formula.
This verifies the constructed instance, not an empirical calibration or a
generic/asymptotic assertion for the whole security-payoff family.
"""
from fractions import Fraction as F
from itertools import permutations, product
from math import isqrt, prod

P = (F(1, 10),) * 4
LOSS = (F(1),) * 4
COST = (F(9, 100), F(252, 3125), F(9, 100), F(391, 6250))
# One-based arcs are stored as (source, receiver).
BASE = {(1, 2), (2, 3), (3, 4), (4, 1)}
Q = {(1, 4), (3, 2)}
TRANSFER = {(4, 1): F(1, 5), (1, 2): F(1, 5),
            (2, 3): F(1, 5), (3, 2): F(1, 5),
            (3, 4): F(2, 5), (1, 4): F(2, 5)}


def pure_cost(i, actions, edges):
    survival = prod(1 - (1 - actions[j - 1]) * TRANSFER[j, i]
                    for j, receiver in edges if receiver == i)
    own = actions[i - 1]
    direct = (1 - own) * P[i - 1]
    return own * COST[i - 1] + LOSS[i - 1] * (
        direct + (1 - direct) * (1 - survival))


def averaged_cost(i, own_action, x, edges):
    others = [j for j in range(1, 5) if j != i]
    total = F(0)
    for choices in product((0, 1), repeat=3):
        actions = [0] * 4
        actions[i - 1] = own_action
        weight = F(1)
        for j, action in zip(others, choices):
            actions[j - 1] = action
            weight *= x[j - 1] if action else 1 - x[j - 1]
        total += weight * pure_cost(i, actions, edges)
    return total


def normalized_gap(i, x, edges):
    return prod(1 - TRANSFER[j, i] + TRANSFER[j, i] * x[j - 1]
                for j, receiver in edges if receiver == i) - (
                    COST[i - 1] / (P[i - 1] * LOSS[i - 1]))


def bilinear_coefficients(i, edges):
    # Receiver 2 or 4: variables x_1 and x_3, in order 1, x, y, xy.
    u = TRANSFER[1, i] if (1, i) in edges else F(0)
    v = TRANSFER[3, i] if (3, i) in edges else F(0)
    r = COST[i - 1] / (P[i - 1] * LOSS[i - 1])
    return ((1 - u) * (1 - v) - r, u * (1 - v),
            (1 - u) * v, u * v)


def rational_sqrt(value):
    n, d = isqrt(value.numerator), isqrt(value.denominator)
    assert n * n == value.numerator and d * d == value.denominator
    return F(n, d)


def solve_all(edges):
    a, b, c, d = bilinear_coefficients(2, edges)
    e, f, g, h = bilinear_coefficients(4, edges)
    if c == d == 0:
        assert b != 0
        x = -a / b
        assert g + h * x != 0
        pairs = [(x, -(e + f * x) / (g + h * x))]
    elif f == h == 0:
        assert g != 0
        y = -e / g
        assert b + d * y != 0
        pairs = [(-(a + c * y) / (b + d * y), y)]
    else:
        # Eliminate y: (e+fx)(c+dx) - (g+hx)(a+bx) = 0.
        A = f * d - h * b
        B = e * d + f * c - g * b - h * a
        C = e * c - g * a
        assert A != 0
        discriminant = B * B - 4 * A * C
        assert discriminant > 0
        square_root = rational_sqrt(discriminant)
        xs = [(-B - square_root) / (2 * A),
              (-B + square_root) / (2 * A)]
        # No lost component at the elimination denominator's zero.
        assert d != 0 and a + b * (-c / d) != 0
        pairs = [(x, -(a + b * x) / (c + d * x)) for x in xs]
    return sorted((x, F(1, 2), y, F(1, 2)) for x, y in pairs)


def jacobian(x, edges):
    matrix = []
    for i in range(1, 5):
        incoming = sorted(j for j, receiver in edges if receiver == i)
        matrix.append([
            TRANSFER[j, i] * prod(
                1 - TRANSFER[k, i] + TRANSFER[k, i] * x[k - 1]
                for k in incoming if k != j) if j in incoming else F(0)
            for j in range(1, 5)])
    return matrix


def determinant(matrix):
    total = F(0)
    for order in permutations(range(4)):
        inversions = sum(order[i] > order[j]
                         for i in range(4) for j in range(i + 1, 4))
        total += (-1) ** inversions * prod(matrix[i][order[i]]
                                          for i in range(4))
    return total


def main():
    cases = [('E', BASE, [(F(4, 125), F(1, 2), F(8, 125), F(1, 2))]),
             ('E+14', BASE | {(1, 4)},
              [(F(4, 125), F(1, 2), F(403, 383), F(1, 2))]),
             ('E+32', BASE | {(3, 2)},
              [(F(122, 127), F(1, 2), F(8, 125), F(1, 2))]),
             ('H', BASE | Q,
              [(F(1, 5), F(1, 2), F(4, 5), F(1, 2)),
               (F(4, 5), F(1, 2), F(1, 5), F(1, 2))])]
    mixed_gap_checks = 0
    root_checks = 0
    for name, edges, expected in cases:
        for i in range(1, 5):
            incoming_sum = sum(TRANSFER[j, i] for j, receiver in edges
                               if receiver == i)
            assert 0 < P[i - 1] < 1
            assert 0 < COST[i - 1] < LOSS[i - 1]
            assert P[i - 1] + incoming_sum < 1
        for actions in product((0, 1), repeat=4):
            for i in range(1, 5):
                cost = pure_cost(i, actions, edges)
                assert 0 <= cost <= COST[i - 1] + LOSS[i - 1]
        for x in product((F(1, 5), F(1, 2), F(4, 5)), repeat=4):
            for i in range(1, 5):
                direct_gap = (averaged_cost(i, 1, x, edges)
                              - averaged_cost(i, 0, x, edges))
                assert direct_gap == -P[i - 1] * LOSS[i - 1] * (
                    normalized_gap(i, x, edges))
                mixed_gap_checks += 1
        roots = solve_all(edges)
        assert roots == expected
        for x in roots:
            assert all(normalized_gap(i, x, edges) == 0 for i in range(1, 5))
            assert determinant(jacobian(x, edges)) != 0
            if all(0 < value < 1 for value in x):
                for i in range(1, 5):
                    assert averaged_cost(i, 1, x, edges) == averaged_cost(
                        i, 0, x, edges)
            root_checks += 1
        interior_count = sum(all(0 < value < 1 for value in x) for x in roots)
        print(f'{name}: isolated complex roots={len(roots)}, '
              f'totally mixed equilibria={interior_count}')
    for x, expected in [(cases[-1][2][0], F(-6, 625)),
                        (cases[-1][2][1], F(6, 625))]:
        a, b, c, d = bilinear_coefficients(2, BASE | Q)
        e, f, g, h = bilinear_coefficients(4, BASE | Q)
        reduced_det = ((b + d * x[2]) * (g + h * x[0])
                       - (c + d * x[0]) * (f + h * x[2]))
        assert reduced_det == expected
    print(f'Passed {mixed_gap_checks} independent pure-profile averaging checks '
          f'and {root_checks} exact root/regularity checks.')


if __name__ == '__main__':
    main()
