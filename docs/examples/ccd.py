"""Automatic derivation of CCD equations.

"""

import pickle

from pyspark import SparkConf, SparkContext
from sympy import IndexedBase, Rational

from drudge import PartHoleDrudge, CR, AN

conf = SparkConf().setAppName('CCSD-derivation')
ctx = SparkContext(conf=conf)
dr = PartHoleDrudge(ctx)
p = dr.names

c_ = dr.op[AN]
c_dag = dr.op[CR]
a, b = p.V_dumms[:2]
i, j = p.O_dumms[:2]

t = IndexedBase('t')
dr.set_dbbar_base(t, 2)

doubles = dr.sum(
    (a, p.V), (b, p.V), (i, p.O), (j, p.O),
    t[a, b, i, j] * c_dag[a] * c_dag[b] * c_[j] * c_[i]
)

curr = dr.ham
h_bar = dr.ham
for i in range(0, 4):
    curr = (curr | doubles).simplify() * Rational(1, i + 1)
    h_bar += curr

en_eqn = dr.eval_fermi_vev(h_bar)

proj = c_dag[i] * c_dag[j] * c_[b] * c_[a]
t2_eqn = dr.eval_fermi_vev(proj * h_bar)

with open('ccd_eqns.pickle') as fp:
    pickle.dump([en_eqn, t2_eqn], fp)
