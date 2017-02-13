"""Tests for the general model with explicit one-half spin."""

import pytest

from sympy import IndexedBase, symbols, Rational

from drudge import CR, AN, UP, DOWN, SpinOneHalfGenDrudge


@pytest.fixture(scope='module')
def dr(spark_ctx):
    """The fixture with a general spin one-half drudge."""
    return SpinOneHalfGenDrudge(spark_ctx)


def test_spin_one_half_general_drudge_has_properties(dr):
    """Test the basic properties of the drudge."""

    assert dr.spin_vals == [UP, DOWN]
    assert dr.orig_ham.n_terms == 2 + 4
    assert dr.ham.n_terms == 2 + 3


def test_restricted_hf_theory(dr):
    """Test the derivation of restricted HF theory."""

    p = dr.names

    c_dag = p.c_dag
    c_ = p.c_
    a, b, c, d = p.L_dumms[:4]
    alpha = symbols('alpha')

    # Concrete summation.
    rot = dr.sum(
        (alpha, [UP, DOWN]), Rational(1, 2) * c_dag[a, alpha] * c_[b, alpha]
    )

    comm = (dr.ham | rot).simplify()

    # Restricted theory has same density for spin up and down.
    rho = IndexedBase('rho')
    res = comm.eval_vev(lambda op1, op2, _: (
        rho[op2.indices[1], op1.indices[1]]
        if op1.indices[0] == CR and op2.indices[0] == AN
           and op1.indices[2] == op2.indices[2]
        else 0
    )).simplify()

    # The expected result.
    t = dr.one_body
    u = dr.two_body

    f = IndexedBase('f')
    expected = dr.einst(rho[b, c] * f[c, a] - f[b, c] * rho[c, a])
    expected = expected.subst(f[a, b], dr.einst(
        t[a, b] +
        2 * u[a, c, b, d] * rho[d, c] - u[c, a, b, d] * rho[d, c]
    ))
    expected = expected.simplify()

    assert res == expected
