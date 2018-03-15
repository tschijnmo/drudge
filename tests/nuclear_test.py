"""Tests for special utilities related to nuclear problems."""

import pytest
from sympy import Symbol, simplify, latex, symbols, KroneckerDelta

from drudge import NuclearBogoliubovDrudge, Range
from drudge.nuclear import JOf, TildeOf, MOf, NOf, LOf, PiOf, TOf, CG


@pytest.fixture(scope='module')
def nuclear(spark_ctx):
    """Set up the drudge to test."""
    return NuclearBogoliubovDrudge(spark_ctx)


def test_qn_accessors():
    """Test the symbolic functions for quantum number access."""

    k = Symbol('k')
    for acc in [JOf, TildeOf, MOf, NOf, LOf, PiOf, TOf]:
        # Test that they are considered integers.
        e = acc(k)
        assert simplify((-1) ** (e * 2)) == 1

        latex_form = latex(e)
        assert latex_form[-3:] == '{k}'


def test_jm_dummies_are_integers(nuclear: NuclearBogoliubovDrudge):
    """Test that the angular momentum dummies has the right assumptions."""
    p = nuclear.names
    for i in [p.m1, p.m2, p.M1, p.M2, p.J1, p.J2]:
        assert simplify((-1) ** (i * 2)) == 1


def test_varsh_872_4(nuclear: NuclearBogoliubovDrudge):
    """Test simplification based on Varshalovich 8.7.2 Eq (4)."""
    dr = nuclear
    c, gamma, c_prm, gamma_prm = symbols('c gamma cprm gammaprm')
    a, alpha, b, beta = symbols('a alpha b beta')

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
        res = tensor.simplify_cg()
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
        'a alpha b beta bprm betaprm'
    )
    c, gamma = symbols('c gamma')
    sums = [
        (alpha, Range('m', -a, a + 1)),
        (gamma, Range('M', -c, c + 1))
    ]
    amp = CG(a, alpha, b, beta, c, gamma) * CG(
        a, alpha, b_prm, beta_prm, c, gamma
    )

    expected = dr.sum(
        KroneckerDelta(b, b_prm) * KroneckerDelta(beta, beta_prm)
        * (2 * c + 1) / (2 * b + 1)
    )
    for sums_i in [sums, reversed(sums)]:
        tensor = dr.sum(*sums_i, amp)
        assert (tensor.simplify_cg() - expected).simplify() == 0
