"""Automatic derivation of different CC working equations.

The purpose of this sample script is to demonstrate that tensors can be created
by programming when it is impossible, or less desirable, to create them
statically with the mathematical notation.  Drudge is a Python library.  Users
have the full power of Python at hand.
"""

import argparse
import collections
import urllib.request

from pyspark import SparkConf, SparkContext
from sympy import IndexedBase, Rational, factorial

from drudge import PartHoleDrudge, sum_, prod_, Stopwatch

#
# Job preparation
#

parser = argparse.ArgumentParser()
parser.add_argument(
    'theory', choices=['ccd', 'ccsd', 'ccsdt', 'ccsdtq'],
    help='The CC theory to derive.'
)
args = parser.parse_args()
theory = args.theory

conf = SparkConf().setAppName('{}-derivation'.format(theory))
ctx = SparkContext(conf=conf)

#
# Setting input tensors
#

dr = PartHoleDrudge(ctx)
dr.full_simplify = False

print('Derive {} theory, default partition {}'.format(
    theory.upper(), dr.num_partitions
))
p = dr.names

c_ = p.c_
c_dag = p.c_dag
v_dumms = p.V_dumms
o_dumms = p.O_dumms

ORDER_OF = {'s': 1, 'd': 2, 't': 3, 'q': 4}

t = IndexedBase('t')
orders = []
for i in theory[2:]:
    order = ORDER_OF[i]
    orders.append(order)
    if order > 1:
        dr.set_dbbar_base(t, order)

corr = dr.einst(sum_(
    Rational(1, factorial(i) ** 2) *
    t[tuple(v_dumms[:i]) + tuple(o_dumms[:i])] *
    prod_(c_dag[j] for j in v_dumms[:i]) *
    prod_(c_[j] for j in reversed(o_dumms[:i]))
    for i in orders
))

print('Problem setting up done.')

#
# Similarity transform the Hamiltonian
#

stopwatch = Stopwatch()

curr = dr.ham
h_bar = dr.ham
for i in range(4):
    curr = (curr | corr).simplify() / (i + 1)
    stopwatch.tock('Commutator order {}'.format(i + 1), curr)
    h_bar += curr
    continue

h_bar = h_bar.simplify()
h_bar.repartition(cache=True)
n_terms = h_bar.n_terms

stopwatch.tock('H-bar assembly', h_bar)

en_eqn = h_bar.eval_fermi_vev().simplify()
stopwatch.tock('Energy equation', en_eqn)

dr.wick_parallel = 1

amp_eqns = collections.OrderedDict()
for order in orders:
    proj = prod_(
        c_dag[j] for j in o_dumms[:order]
    ) * prod_(
        c_[j] for j in reversed(v_dumms[:order])
    )

    eqn = (proj * h_bar).eval_fermi_vev().simplify()
    stopwatch.tock('T{} equation'.format(order), eqn)
    amp_eqns[order] = eqn

    continue

stopwatch.tock_total()

# Check with the result from TCE.
TCE_BASE_URL = 'http://www.scs.illinois.edu/~sohirata/'
tce_labels = ['e']
tce_labels.extend('t{}'.format(i) for i in orders)
tce_files = ('{}_{}.out'.format(theory, i) for i in tce_labels)

tce_res = [
    dr.parse_tce(
        urllib.request.urlopen(TCE_BASE_URL + i).read().decode(),
        {i: t for i in orders}
    ).simplify()
    for i in tce_files
    ]

print('Checking with TCE result: ')
print('Energy: ', en_eqn == tce_res[0])
for i, order in enumerate(amp_eqns.keys()):
    diff = (amp_eqns[order] - tce_res[i + 1]).simplify()
    print('T{} amplitude: '.format(order), diff == 0)
