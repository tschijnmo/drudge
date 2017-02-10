"""Automatic derivation of CCSDTQ equations.

The purpose of this sample script is to demonstrate that tensors can be created
by programming when it is less desirable to create them statically with the
mathematical notation.  Drudge is a Python library.  Users have the full power
of Python at hand.
"""

import os
import time
import urllib.request

from pyspark import SparkConf, SparkContext
from sympy import IndexedBase, Rational, factorial

from drudge import PartHoleDrudge

n_cpus = os.cpu_count()
if 'SLURM_JOB_NUM_NODES' in os.environ:
    n_cpus *= int(os.environ['SLURM_JOB_NUM_NODES'])
n_parts = n_cpus

conf = SparkConf().setAppName('CCSDTQ-derivation')
ctx = SparkContext(conf=conf)
dr = PartHoleDrudge(ctx)
p = dr.names

c_ = p.c_
c_dag = p.c_dag
v_dumms = p.V_dumms
o_dumms = p.O_dumms

ORDER = 4

cluster_bases = {}
for i in range(ORDER):
    order = i + 1
    t = IndexedBase('t{}'.format(order))
    if order > 1:
        dr.set_dbbar_base(t, order)
    cluster_bases[order] = t

cluster_ops = []
for i, v in cluster_bases.items():
    op = Rational(1, factorial(i) ** 2) * v[
        tuple(v_dumms[:i]) + tuple(o_dumms[:i])
        ]
    for j in v_dumms[:i]:
        op = op * c_dag[j]
    for j in reversed(o_dumms[:i]):
        op = op * c_[j]
    cluster_ops.append(op)
    continue

corr = dr.einst(sum(cluster_ops))

print('Problem setting up done.')
time_begin = time.time()

curr = dr.ham
h_bar = dr.ham
for i in range(4):
    curr = (curr | corr).simplify(n_parts) * Rational(1, i + 1)
    curr.cache()
    h_bar += curr
h_bar.repartition(n_parts, cache=True)

now = time.time()
print('Similarity-transformed hamiltonian done.  wall time: {}'.format(
    now - time_begin
))
time_begin = now

en_eqn = h_bar.eval_fermi_vev().simplify(n_parts)

amp_eqns = []
for i in range(ORDER):
    order = i + 1

    proj = 1
    for j in o_dumms[:order]:
        proj = proj * c_dag[j]
    for j in reversed(v_dumms[:order]):
        proj = proj * c_[j]

    eqn = (proj * h_bar).eval_fermi_vev().simplify(n_parts)
    amp_eqns.append(eqn)

now = time.time()
print('CC equation derivation done.  wall time: {}'.format(
    now - time_begin
))

# Check with the result from TCE.
TCE_BASE_URL = 'http://www.scs.illinois.edu/~sohirata/'
tce_res = [
    dr.parse_tce(
        urllib.request.urlopen(TCE_BASE_URL + i).read().decode(),
        cluster_bases
    ).simplify()
    for i in [
        'ccsdtq_e.out', 'ccsdtq_t1.out', 'ccsdtq_t2.out',
        'ccsdtq_t3.out', 'ccsdtq_t4.out',
    ]
    ]

print('Checking with TCE result: ')
print('Energy: ', en_eqn == tce_res[0])
order = 1
for i, j in zip(amp_eqns, tce_res[1:]):
    print('T{} amplitude: '.format(order), i == j)
    order += 1
    continue
