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


def test_tensor_can_be_created(free_alg):
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


def test_tensor_has_basic_operations(free_alg):
    """Test some of the basic operations on tensors.

    Tested in this module:

        1. Addition.
        2. Merge.
        3. Free variable.
        4. Dummy reset.
        5. Equality comparison.
    """

    dr, p = free_alg
    i, j, k, l, m = p.dumms[:5]
    x = IndexedBase('x')
    r = p.r
    v = p.v
    tensor = (
        dr.sum((l, r), x[i, l] * v[l]) +
        dr.sum((m, r), x[j, m] * v[m])
    )

    # Without dummy resetting, they cannot be merged.
    assert tensor.n_terms == 2
    assert tensor.merge().n_terms == 2

    # Free variables are important for dummy resetting.
    free_vars = tensor.free_vars
    assert free_vars == {x.label, i, j}

    # Reset dummy.
    reset = tensor.reset_dumms()
    expected = (
        dr.sum((k, r), x[i, k] * v[k]) +
        dr.sum((k, r), x[j, k] * v[k])
    )
    assert reset == expected
    assert reset.local_terms == expected.local_terms

    # Merge the terms.
    merged = reset.merge()
    assert merged.n_terms == 1
    term = merged.local_terms[0]
    assert term == Term([(k, r)], x[i, k] + x[j, k], [v[k]])
