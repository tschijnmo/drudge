"""Tests for the basic tensor facilities using free algebra."""

import types

import pytest
from pyspark import SparkContext, SparkConf
from sympy import sympify, IndexedBase, sin, cos, KroneckerDelta, symbols

from drudge import Drudge, Range, Vec, Term, Perm, NEG


@pytest.fixture(scope='module')
def free_alg():
    """Initialize the environment for a free algebra."""

    conf = SparkConf().setMaster('local[2]').setAppName('free-alegra')
    ctx = SparkContext(conf=conf)
    dr = Drudge(ctx)

    r = Range('R')
    dumms = sympify('i, j, k, l, m, n')
    dr.set_dumms(r, dumms)

    s = Range('S')
    s_dumms = symbols('alpha beta')
    dr.set_dumms(s, s_dumms)

    v = Vec('v')

    m = IndexedBase('m')
    dr.set_symm(m, Perm([1, 0], NEG))

    return dr, types.SimpleNamespace(
        r=r, dumms=dumms, s=s, s_dumms=s_dumms, v=v, m=m
    )


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


def test_tensor_can_be_simplified_amp(free_alg):
    """Test the amplitude simplification for tensors.

    More than trivial tensor amplitude simplification is tested here.  Currently
    it mostly concentrates on the dispatching to SymPy and delta simplification.
    The master simplification is also tested.
    """

    dr, p = free_alg
    r = p.r
    s = p.s
    v = p.v
    i, j = p.dumms[:2]
    alpha = p.s_dumms[0]

    x = IndexedBase('x')
    y = IndexedBase('y')
    theta = sympify('theta')

    tensor = (
        dr.sum((i, r), sin(theta) ** 2 * x[i] * v[i]) +
        dr.sum(
            (i, r), (j, r),
            cos(theta) ** 2 * x[j] * KroneckerDelta(i, j) * v[i]
        ) +
        dr.sum((i, r), (alpha, s), KroneckerDelta(i, alpha) * y[i] * v[i])
    )
    assert tensor.n_terms == 3

    first = tensor.simplify_amps()
    # Now we should have one term killed.
    assert first.n_terms == 2

    # Merge again should really simplify.
    merged = first.reset_dumms().merge().simplify_amps()
    assert merged.n_terms == 1
    expected = dr.sum((i, r), x[i] * v[i])
    assert merged == expected

    # The master simplification should do it in one turn.
    simpl = tensor.simplify()
    assert simpl == expected


def test_tensor_can_be_canonicalized(free_alg):
    """Test tensor canonicalization in simplification.

    The master simplification function is tested, the core simplification is at
    the canonicalization.  Equality testing with zero is also tested.
    """

    dr, p = free_alg
    i, j = p.dumms[:2]
    r = p.r
    m = p.m
    v = p.v

    tensor = (
        dr.sum((i, r), (j, r), m[i, j] * v[i] * v[j]) +
        dr.sum((i, r), (j, r), m[j, i] * v[i] * v[j])
    )
    assert tensor.n_terms == 2

    res = tensor.simplify()
    assert res == 0
