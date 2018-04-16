"""Tests for special utilities related to nuclear problems."""

import random

import pytest
from sympy import Symbol, latex, symbols, KroneckerDelta, sqrt

from drudge import NuclearBogoliubovDrudge, Range, Term
from drudge.nuclear import (
    JOf, MOf, CG, Wigner6j, Wigner3j, _simpl_pono_term
)


@pytest.fixture(scope='module')
def nuclear(spark_ctx):
    """Set up the drudge to test."""
    return NuclearBogoliubovDrudge(spark_ctx)


#
# Test of the core power of negative one simplification.
#

def test_jm_acc_as_half_integer():
    """Test j/m access of single-particle k to be half integers.
    """

    k = Symbol('k')
    for acc in [JOf, MOf]:
        e = acc(k)
        term = Term(
            (), (-1) ** (e * 2), ()
        )
        res = _simpl_pono_term(term, [])

        assert len(res.sums) == 0
        assert res.amp == -1
        assert len(res.vecs) == 0


def test_coll_jm_integer(nuclear: NuclearBogoliubovDrudge):
    """Test integrity of collective angular momentum symbols.
    """

    p = nuclear.names
    k = Symbol('k')
    wigner = Wigner3j(p.J1, p.M1, p.J2, p.M2, JOf(k), p.m1)
    for factor, phase in [
        (p.M1, 1), (p.M2, 1), (p.J1, 1), (p.J2, 1),
        (JOf(k), -1), (p.m1, -1)
    ]:
        term = Term(
            (), (-1) ** (factor * 2) * wigner, ()
        )
        res = _simpl_pono_term(term, nuclear.resolvers.value)

        assert len(res.sums) == 0
        assert res.amp == phase * wigner
        assert len(res.vecs) == 0


def test_wigner_3j_m_rels_simpl():
    """Test simplification based on m-sum rules of Wigner 3j symbols.
    """
    j = Symbol('j')
    a, b, c, d, e = symbols('a b c d e')
    wigner_3js = Wigner3j(j, a, j, b, j, c) * Wigner3j(j, c, j, d, j, e)

    for amp in [
        (-1) ** (a + b + c),
        (-1) ** (c + d + e),
        (-1) ** (a + b + 2 * c + d + e),
        (-1) ** (a + b - d - e)
    ]:
        term = Term((), wigner_3js * amp, ())
        res = _simpl_pono_term(term, [])
        assert len(res.sums) == 0
        assert res.amp == wigner_3js
        assert len(res.vecs) == 0


def test_varsh_872_4(nuclear: NuclearBogoliubovDrudge):
    """Test simplification based on Varshalovich 8.7.2 Eq (4)."""
    dr = nuclear
    c, gamma, c_prm, gamma_prm = symbols('c gamma cprm gammaprm', integer=True)
    a, alpha, b, beta = symbols('a alpha b beta', integer=True)

    m_range = Range('m')
    sums = [
        (alpha, m_range[-a, a + 1]), (beta, m_range[-b, b + 1])
    ]
    amp = CG(a, alpha, b, beta, c, gamma) * CG(
        a, alpha, b, beta, c_prm, gamma_prm
    )

    # Make sure that the pattern matching works in any way the summations are
    # written.
    for sums_i in [sums, reversed(sums)]:
        tensor = dr.sum(*sums_i, amp)
        res = tensor.simplify_am()
        assert res.n_terms == 1
        term = res.local_terms[0]
        assert len(term.sums) == 0
        assert term.amp == KroneckerDelta(
            c, c_prm
        ) * KroneckerDelta(gamma, gamma_prm)


def test_varsh_872_5(nuclear: NuclearBogoliubovDrudge):
    """Test simplification based on the rule in Varshalovich 8.7.2 Eq (5).
    """
    dr = nuclear
    a, alpha, b, beta, b_prm, beta_prm = symbols(
        'a alpha b beta bprm betaprm', integer=True
    )
    c, gamma = symbols('c gamma', integer=True)
    sums = [
        (alpha, Range('m', -a, a + 1)),
        (gamma, Range('M', -c, c + 1))
    ]
    amp = CG(a, alpha, b, beta, c, gamma) * CG(
        a, alpha, b_prm, beta_prm, c, gamma
    )

    expected = (
            KroneckerDelta(b, b_prm) * KroneckerDelta(beta, beta_prm)
            * (2 * c + 1) / (2 * b + 1)
    )
    for sums_i in [sums, reversed(sums)]:
        tensor = dr.sum(*sums_i, amp)
        res = tensor.deep_simplify().merge()
        assert res.n_terms == 1
        term = res.local_terms[0]
        assert len(term.sums) == 0
        assert len(term.vecs) == 0
        assert (term.amp - expected).simplify() == 0


def test_varsh_911_8(nuclear: NuclearBogoliubovDrudge):
    """Test simplification based on the rule in Varshalovich 9.1.1 Eq (8).
    """
    dr = nuclear
    j, m, j12, m12, j2, m2, j1, m1, j_prm, m_prm, j23, m23, j3, m3 = symbols(
        'j m j12 m12 j2 m2 j1 m1 jprm mprm j23 m23 j3 m3', integer=True
    )
    m_range = Range('m')
    sums = [(m_i, m_range[-j_i, j_i + 1]) for m_i, j_i in [
        (m1, j1), (m2, j2), (m3, j3), (m12, j12), (m23, j23)
    ]]
    amp = CG(j12, m12, j3, m3, j, m) * CG(j1, m1, j2, m2, j12, m12) * CG(
        j1, m1, j23, m23, j_prm, m_prm
    ) * CG(j2, m2, j3, m3, j23, m23)

    expected = (
            KroneckerDelta(j, j_prm) * KroneckerDelta(m, m_prm)
            * (-1) ** (j1 + j2 + j3 + j)
            * sqrt(2 * j12 + 1) * sqrt(2 * j23 + 1)
            * Wigner6j(j1, j2, j12, j3, j, j23)
    )

    # For performance reason, just test a random arrangement of the summations.
    random.shuffle(sums)
    tensor = dr.sum(*sums, amp)
    res = tensor.deep_simplify().merge()
    assert res.n_terms == 1
    term = res.local_terms[0]
    assert len(term.sums) == 0
    assert len(term.vecs) == 0
    assert (term.amp - expected).simplify() == 0


def test_wigner3j_sum_to_wigner6j(nuclear: NuclearBogoliubovDrudge):
    """Test simplification of sum of product of four 3j's to a 6j.

    This test tries to simplify the original LHS of the equation from the
    Wolfram website.
    """

    dr = nuclear
    j1, j2, j3, jprm3, j4, j5, j6 = symbols(
        'j1 j2 j3 jprm3 j4 j5 j6', integer=True
    )
    m1, m2, m3, mprm3, m4, m5, m6 = symbols(
        'm1 m2 m3 mprm3 m4 m5 m6', integer=True
    )

    m_range = Range('m')
    sums = [(m_i, m_range[-j_i, j_i + 1]) for m_i, j_i in [
        (m1, j1), (m2, j2), (m4, j4), (m5, j5), (m6, j6)
    ]]

    phase = (-1) ** (
            j1 + j2 + j4 + j5 + j6 - m1 - m2 - m4 - m5 - m6
    )
    amp = (
            Wigner3j(j2, m2, j3, -m3, j1, m1)
            * Wigner3j(j1, -m1, j5, m5, j6, m6)
            * Wigner3j(j5, -m5, jprm3, mprm3, j4, m4)
            * Wigner3j(j4, -m4, j2, -m2, j6, -m6)
    )

    expected = (
            ((-1) ** (j3 - m3) / (2 * j3 + 1))
            * KroneckerDelta(j3, jprm3) * KroneckerDelta(m3, mprm3)
            * Wigner6j(j1, j2, j3, j4, j5, j6)
    ).expand().simplify()

    # For performance reason, just test a random arrangement of the summations.
    random.shuffle(sums)
    tensor = dr.sum(*sums, phase * amp)
    res = tensor.deep_simplify().merge()
    assert res.n_terms == 1
    term = res.local_terms[0]
    assert len(term.sums) == 0
    assert len(term.vecs) == 0
    assert (term.amp - expected).simplify() == 0
