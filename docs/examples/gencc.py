"""Automatic derivation of different CC working equations.

The purpose of this sample script is to demonstrate that tensors can be created
by programming when it is impossible, or less desirable, to create them
statically with the mathematical notation.  Drudge is a Python library.  Users
have the full power of Python at hand.
"""

import argparse
import collections
import os
import time
import urllib.request

from pyspark import SparkConf, SparkContext
from sympy import IndexedBase, Rational, factorial

from drudge import PartHoleDrudge, sum_, prod_

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

n_cpus = os.cpu_count()
if 'SLURM_JOB_NUM_NODES' in os.environ:
    n_cpus *= int(os.environ['SLURM_JOB_NUM_NODES'])
n_parts = n_cpus * 3

conf = SparkConf().setAppName('{}-derivation'.format(theory))
ctx = SparkContext(conf=conf)

print('Derive {} theory, default partition {}'.format(theory.upper(), n_parts))

#
# Setting input tensors
#

dr = PartHoleDrudge(ctx, num_partitions=n_parts, full_simplify=False)
p = dr.names

c_ = p.c_
c_dag = p.c_dag
v_dumms = p.V_dumms
o_dumms = p.O_dumms

ORDER_OF = {'s': 1, 'd': 2, 't': 3, 'q': 4}

cluster_bases = collections.OrderedDict()
for i in theory[2:]:
    order = ORDER_OF[i]
    t = IndexedBase('t{}'.format(order))
    if order > 1:
        dr.set_dbbar_base(t, order)
    cluster_bases[order] = t

corr = dr.einst(sum_(
    Rational(1, factorial(i) ** 2) *
    v[tuple(v_dumms[:i]) + tuple(o_dumms[:i])] *
    prod_(c_dag[j] for j in v_dumms[:i]) *
    prod_(c_[j] for j in reversed(o_dumms[:i]))
    for i, v in cluster_bases.items()
))

print('Problem setting up done.')

#
# Similarity transform the Hamiltonian
#

time_begin = time.time()

curr = dr.ham
h_bar = dr.ham
for i in range(4):
    curr = (curr | corr).simplify() * Rational(1, i + 1)
    curr.cache()
    h_bar += curr
    n_terms = curr.n_terms

    now = time.time()
    print('Commutator order {} done, {} terms, wall time {}s'.format(
        i + 1, n_terms, now - time_begin
    ))
    time_begin = now

    continue

h_bar = h_bar.simplify().expand()
h_bar.repartition(cache=True)
n_terms = h_bar.n_terms

now = time.time()
print('H-bar assembly done.  {} terms,  wall time {}s'.format(
    n_terms, now - time_begin
))
time_begin = now

en_eqn = h_bar.eval_fermi_vev().simplify()
en_eqn.cache()
n_terms = en_eqn.n_terms
now = time.time()
print('Energy equation done.  {} terms, wall time {}s'.format(
    n_terms, now - time_begin
))
time_begin = now

amp_eqns = collections.OrderedDict()
for order in cluster_bases.keys():
    proj = prod_(
        c_dag[j] for j in o_dumms[:order]
    ) * prod_(
        c_[j] for j in reversed(v_dumms[:order])
    )

    eqn = (proj * h_bar).eval_fermi_vev().simplify()
    eqn.cache()
    n_terms = eqn.n_terms
    amp_eqns[order] = eqn

    now = time.time()
    print('T{} equation done.  {} terms, wall time {}s'.format(
        order, n_terms, now - time_begin
    ))
    time_begin = now

    continue

# Check with the result from TCE.
TCE_BASE_URL = 'http://www.scs.illinois.edu/~sohirata/'
tce_labels = ['e']
tce_labels.extend('t{}'.format(i) for i in cluster_bases.keys())
tce_files = ('{}_{}.out'.format(theory, i) for i in tce_labels)

tce_res = [
    dr.parse_tce(
        urllib.request.urlopen(TCE_BASE_URL + i).read().decode(),
        cluster_bases
    ).simplify()
    for i in tce_files
    ]

print('Checking with TCE result: ')
print('Energy: ', en_eqn == tce_res[0])
for i, order in enumerate(amp_eqns.keys()):
    diff = (amp_eqns[order] - tce_res[i + 1]).simplify()
    print('T{} amplitude: '.format(order), diff == 0)
