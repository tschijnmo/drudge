"""Automatic derivation of resitrict CCDF theory.

This script serves as an example of using drudge for complex symbolic
manipulations.  The derivation here is going to be based on the approach in GE
Scuseria et al, J Chem Phys 89 (1988) 7382 (10.1063/1.455269).

"""

from pyspark import SparkConf, SparkContext
from sympy import IndexedBase, Rational, symbols

from drudge import SpinOneHalfPartHoleDrudge, Vec, UP, DOWN, TimeStamper

# Environment setting up.

conf = SparkConf().setAppName('rccsd')
ctx = SparkContext(conf=conf)
dr = SpinOneHalfPartHoleDrudge(ctx)
dr.full_simplify = False

p = dr.names
c_dag = p.c_dag
c_ = p.c_
a, b, c, d = p.V_dumms[:4]
i, j, k, l = p.O_dumms[:4]

#
# Cluster excitation operator
# 
# Here, we first write the cluster excitation operator in terms of the
# unitary group generator.  Then they will be substituted by their fermion
# operator definition.
#

t1 = IndexedBase('t^1')
t2 = IndexedBase('t^2')
e_ = Vec('E')

cluster_e = dr.einst(
    t1[a, i] * e_[a, i] +
    Rational(1, 2) * t2[a, b, i, j] * e_[a, i] * e_[b, j]
)

cluster = cluster_e.subst(
    e_[a, i], c_dag[a, UP] * c_[i, UP] + c_dag[a, DOWN] * c_[i, DOWN]
)
dr.set_n_body_base(t2, 2)
cluster = cluster.simplify()
cluster.cache()

#
# Similarity transform of the Hamiltonian
# 

stamper = TimeStamper()

curr = dr.ham
h_bar = dr.ham
for order in range(4):
    curr = (curr | cluster).simplify() * Rational(1, order + 1)
    stamper.stamp('Commutator order {}'.format(order + 1), curr)
    h_bar += curr
    continue

h_bar = h_bar.simplify()
h_bar.repartition(cache=True)
stamper.stamp('H-bar assembly', h_bar)

en_eqn = h_bar.eval_fermi_vev().simplify()
stamper.stamp('Energy equation', en_eqn)

dr.wick_parallel = 1

e_dag = Vec(r'E^\dagger')
beta, gamma, u, v = symbols('beta gamma u v')
spin = symbols('spin')
e_dag_def = dr.sum((spin, UP, DOWN), c_dag[i, spin] * c_[a, spin])
projs = [e_dag_def.act(e_dag[a, i], p) for p in [
    e_dag[beta, u],
    e_dag[beta, u] * e_dag[gamma, v]
]]

#
# Dump the result to a simple report.
#

amp_eqns = []
for order, proj in enumerate(projs):
    eqn = (proj * h_bar).eval_fermi_vev().simplify()
    stamper.stamp('T{} equation'.format(order + 1), eqn)
    amp_eqns.append(eqn)
    continue

with dr.report('rCCSD.html', 'restricted CCSD theory') as rep:
    rep.add('Energy equation', en_eqn)
    for i, v in enumerate(amp_eqns):
        rep.add(r'\(T^{}\) amplitude equation'.format(i + 1), v)
        continue
