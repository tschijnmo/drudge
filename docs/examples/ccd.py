"""Automatic derivation of CCD equations.

"""

import urllib.request

from pyspark import SparkConf, SparkContext
from sympy import IndexedBase, Rational

from drudge import PartHoleDrudge, CR, AN

conf = SparkConf().setAppName('CCD-derivation')
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
    Rational(1, 4) * t[a, b, i, j] * c_dag[a] * c_dag[b] * c_[j] * c_[i]
)

curr = dr.ham
h_bar = dr.ham
for order in range(0, 4):
    curr = (curr | doubles).simplify() * Rational(1, order + 1)
    h_bar += curr

en_eqn = dr.eval_fermi_vev(h_bar).simplify()

proj = c_dag[i] * c_dag[j] * c_[b] * c_[a]
t2_eqn = dr.eval_fermi_vev(proj * h_bar).simplify()

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
