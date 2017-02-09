"""Automatic derivation of CCD equations.

This script is an example of running drudge serially.
"""

import urllib.request

from dummy_spark import SparkConf, SparkContext
from sympy import IndexedBase, Rational

from drudge import PartHoleDrudge

conf = SparkConf().setAppName('CCD-derivation')
ctx = SparkContext(conf=conf)
dr = PartHoleDrudge(ctx)
p = dr.names

c_ = p.c_
c_dag = p.c_dag
a, b = p.V_dumms[:2]
i, j = p.O_dumms[:2]

t = IndexedBase('t')
dr.set_dbbar_base(t, 2)

doubles = dr.einst(
    Rational(1, 4) * t[a, b, i, j] * c_dag[a] * c_dag[b] * c_[j] * c_[i]
)

curr = dr.ham
h_bar = dr.ham
for order in range(0, 4):
    curr = (curr | doubles).simplify() * Rational(1, order + 1)
    h_bar += curr

en_eqn = h_bar.eval_fermi_vev().simplify()

proj = c_dag[i] * c_dag[j] * c_[b] * c_[a]
t2_eqn = (proj * h_bar).eval_fermi_vev().simplify()

# Check with the result from TCE.
TCE_BASE_URL = 'http://www.scs.illinois.edu/~sohirata/'
tce_res = [
    dr.parse_tce(
        urllib.request.urlopen(TCE_BASE_URL + i).read().decode(),
        {2: t}
    ).simplify()
    for i in ['ccd_e.out', 'ccd_t2.out']
    ]

print('Checking with TCE result: ')
print('Energy: ', en_eqn == tce_res[0])
print('T2 amplitude: ', t2_eqn == tce_res[1])
