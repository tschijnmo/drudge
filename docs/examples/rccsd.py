"""Automatic derivation of resitrict CCDF theory.

This script serves as an example of using drudge for complex symbolic
manipulations.  The derivation here is going to be based on the approach in GE
Scuseria et al, J Chem Phys 89 (1988) 7382 (10.1063/1.455269).

"""

from pyspark import SparkConf, SparkContext
from sympy import IndexedBase, Rational, symbols

from drudge import RestrictedPartHoleDrudge, Stopwatch

# Environment setting up.

conf = SparkConf().setAppName('rccsd')
ctx = SparkContext(conf=conf)
dr = RestrictedPartHoleDrudge(ctx)
dr.full_simplify = False

p = dr.names
e_ = p.e_
a, b, c, d = p.V_dumms[:4]
i, j, k, l = p.O_dumms[:4]

#
# Cluster excitation operator
# 
# Here, we first write the cluster excitation operator in terms of the
# unitary group generator.  Then they will be substituted by their fermion
# operator definition.
#

t = IndexedBase('t')

cluster = dr.einst(
    t[a, i] * e_[a, i] +
    Rational(1, 2) * t[a, b, i, j] * e_[a, i] * e_[b, j]
)

dr.set_n_body_base(t, 2)
cluster = cluster.simplify()
cluster.cache()

#
# Similarity transform of the Hamiltonian
# 

stopwatch = Stopwatch()

curr = dr.ham
h_bar = dr.ham
for order in range(4):
    curr = (curr | cluster).simplify() * Rational(1, order + 1)
    stopwatch.tock('Commutator order {}'.format(order + 1), curr)
    h_bar += curr
    continue

h_bar = h_bar.simplify()
h_bar.repartition(cache=True)
stopwatch.tock('H-bar assembly', h_bar)

en_eqn = h_bar.eval_fermi_vev().simplify()
stopwatch.tock('Energy equation', en_eqn)

dr.wick_parallel = 1

beta, gamma, u, v = symbols('beta gamma u v')
projs = [
    e_[u, beta],
    e_[u, beta] * e_[v, gamma]
]

#
# Dump the result to a simple report.
#

amp_eqns = []
for order, proj in enumerate(projs):
    eqn = (proj * h_bar).eval_fermi_vev().simplify()
    stopwatch.tock('T{} equation'.format(order + 1), eqn)
    amp_eqns.append(eqn)
    continue

stopwatch.tock_total()

with dr.report('rCCSD.html', 'restricted CCSD theory') as rep:
    rep.add('Energy equation', en_eqn)
    for i, v in enumerate(amp_eqns):
        rep.add(r'\(T^{}\) amplitude equation'.format(i + 1), v)
        continue
