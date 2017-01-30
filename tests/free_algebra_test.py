"""Tests for the basic tensor facilities using free algebra."""

import types

import pytest
from pyspark import SparkContext, SparkConf
from sympy import sympify, IndexedBase

from drudge import Drudge, Range, Vec, Term


@pytest.fixture(scope='module')
def free_alg():
    """Initialize the environment for a free algebra."""

    conf = SparkConf().setMaster('local[2]').setAppName('free-alegra')
    ctx = SparkContext(conf=conf)
    dr = Drudge(ctx)

    r = Range('R')
    dumms = sympify('i, j, k, l, m, n')
    dr.set_dumms(r, dumms)

    v = Vec('v')

    return dr, types.SimpleNamespace(r=r, dumms=dumms, v=v)


def test_tensor_creation(free_alg):
    """Test simple tensor creation."""

    dr, p = free_alg
    i = p.dumms[0]
    x = IndexedBase('x')
    tensor = dr.sum((i, p.r), x[i] * p.v[i])

    assert tensor.n_terms == 1

    terms = tensor.local_terms
    assert len(terms) == 1
    term = terms[0]
    assert term == Term([(i, p.r)], x[i], [p.v[i]])
