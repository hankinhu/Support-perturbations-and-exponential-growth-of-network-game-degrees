"""Reproduce the sparse four-player example using only Python's standard library."""
from collections import defaultdict
from fractions import Fraction as Q
from math import comb, pi, sqrt
from decimal import Decimal, localcontext
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
K = (3, 6, 3, 6)
NEIGHBORS = ((1, 2), (2, 3), (3, 0), (0, 1))  # receiver rows, zero-based sources
P = ((0, 2, 1, 0), (0, 0, 2, 4), (1, 0, 0, 2), (2, 4, 0, 0))
Z = (1, 2, 1, 2)

assert tuple(map(sum, P)) == K
assert tuple(sum(P[i][j] for i in range(4)) for j in range(4)) == K
for i in range(4):
    for j in NEIGHBORS[i]:
        assert Q(P[i][j]) == Q(K[i]*Z[j], sum(Z[h] for h in NEIGHBORS[i]))

sigma = [[sum(Q(P[i][j])*(j == h) - Q(P[i][j]*P[i][h], K[i])
              for i in range(4)) for h in range(4)] for j in range(4)]
a, b, c = [row[:3] for row in sigma[:3]]
det = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
assert det == Q(32, 9)
cap = Q(3**18, 2**12)

def exact(m):
    return sum(comb(3*m, t)**2 * comb(6*m, t)**2 for t in range(3*m+1))

def coefficient_dp(m):
    limits = tuple(m*k for k in K)
    states = {(0, 0, 0, 0): 1}
    for i, ki in enumerate(limits):
        for _ in range(ki):
            nxt = defaultdict(int)
            for exp, count in states.items():
                for j in NEIGHBORS[i]:
                    if exp[j] < limits[j]:
                        e = list(exp)
                        e[j] += 1
                        nxt[tuple(e)] += count
            states = nxt
    return states.get(limits, 0)

for m in (1, 2, 3):
    assert coefficient_dp(m) == exact(m)
    for t in range(3*m+1):
        table = ((0,t,3*m-t,0),(0,0,t,6*m-t),(3*m-t,0,0,t),(t,6*m-t,0,0))
        assert tuple(map(sum,table)) == tuple(m*k for k in K)
        assert tuple(sum(table[i][j] for i in range(4)) for j in range(4)) == tuple(m*k for k in K)

rows, records = [], []
for m in (1, 2, 5, 10, 20):
    d = exact(m)
    with localcontext() as ctx:
        ctx.prec = 60
        approximation = Decimal(3) / (Decimal(16)*(Decimal(str(pi))*m)**Decimal('1.5')) * (Decimal(cap.numerator)/cap.denominator)**m
        ratio = Decimal(d)/approximation
        missing = Decimal(1+comb(6*m,3*m)**2)/Decimal(d)
        scientific = f'{Decimal(d):.7E}'
        mantissa, exponent = scientific.split('E')
        shown = str(d) if m <= 2 else mantissa + r'\times10^{' + str(int(exponent)) + '}'
        mantissa_missing, exponent_missing = f'{missing:.3E}'.split('E')
        shown_missing = mantissa_missing + r'\times10^{' + str(int(exponent_missing)) + '}'
        rows.append(f'{m} & ${shown}$ & {ratio:.6f} & ${shown_missing}$ '+r'\\')
        records.append(dict(m=m, exact_degree=str(d), asymptotic_ratio=str(ratio), missing_support_fraction=str(missing)))

table_start = r'''\begin{tabular}{r r r r}
\hline
$m$ & $\mathcal D(F,m\mathbf k)$ & $\mathcal D(F,m\mathbf k)/A_m$ & $1-\mu_{m\mathbf k}(F)/\mathcal D(F,m\mathbf k)$\\
\hline
'''
(ROOT/'sparse-example-table.tex').write_text(table_start+'\n'.join(rows)+'\n'+r'\hline'+'\n'+r'\end{tabular}'+'\n', encoding='utf-8')
(ROOT/'sparse-example-verification.json').write_text(json.dumps(dict(capacity=str(cap), reduced_covariance=[[str(v) for v in row[:3]] for row in sigma[:3]], determinant=str(det), checks='Exact rational margins/scaling/covariance; independent coefficient DP for m=1,2,3', values=records), indent=2), encoding='utf-8')
print(json.dumps(records, indent=2))
print('All exact arithmetic and independent coefficient checks passed.')
