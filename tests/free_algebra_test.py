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
    dr.set_name(v)

    m = IndexedBase('m')
    dr.set_symm(m, Perm([1, 0], NEG))

    return dr, types.SimpleNamespace(
        r=r, dumms=dumms, s=s, s_dumms=s_dumms, v=v, m=m
    )


def test_drudge_has_names(free_alg):
    """Test the name archive for drudge objects.

    Here selected names are tested to makes sure all the code are covered.
    """

    free_alg = free_alg[0]
    p = free_alg.names

    # Range and dummy related.
    assert p.R == Range('R')
    assert len(p.R_dumms) == 6
    assert p.R_dumms[0] == p.i
    assert p.R_dumms[-1] == p.n

    # Vector bases.
    assert p.v == Vec('v')

    # Scalar bases.
    assert p.m == IndexedBase('m')


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
        6. Expansion
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

    # Slightly separate test for expansion.
    c, d = symbols('c d')
    tensor = dr.sum((i, r), x[i] * (c + d) * v[i])
    assert tensor.n_terms == 1
    expanded = tensor.expand()
    assert expanded.n_terms == 2

    # Here we also test concrete summation facility.
    expected = dr.sum(
        (i, r), (j, [c, d]), x[i] * j * v[i]
    )
    assert set(expected.local_terms) == set(expected.local_terms)


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


def test_tensor_math_ops(free_alg):
    """Test tensor math operations.

    Mainly here we test addition and multiplication.
    """

    dr, p = free_alg
    r = p.r
    v = p.v
    w = Vec('w')
    x = IndexedBase('x')
    i, j, k = p.dumms[:3]
    a = sympify('a')

    v1 = dr.sum((i, r), x[i] * v[i])
    w1 = dr.sum((i, r), x[i] * w[i])
    assert v1.n_terms == 1
    assert w1.n_terms == 1

    v1_1 = v1 + 2
    assert v1_1.n_terms == 2
    assert v1_1 == 2 + v1

    w1_1 = w1 + a
    assert w1_1.n_terms == 2
    assert w1_1 == a + w1

    prod = v1_1 * w1_1
    # Test scalar multiplication here as well.
    expected = (
        2 * a + a * v1 + 2 * w1 +
        dr.sum((i, r), (j, r), x[i] * x[j] * v[i] * w[j])
    )
    assert prod.simplify() == expected.simplify()
