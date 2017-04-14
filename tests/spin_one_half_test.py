"""Tests for the drudges with explicit one-half spin."""

import pytest
from sympy import IndexedBase, symbols, Rational, KroneckerDelta, Integer

from drudge import (
    CR, AN, UP, DOWN, SpinOneHalfGenDrudge, SpinOneHalfPartHoleDrudge,
    RestrictedPartHoleDrudge
)


def test_up_down_enum_symbs():
    """Test the desired mathematical properties of enumeration symbols."""

    for i in [UP, DOWN]:
        assert KroneckerDelta(i, i) == 1

    assert KroneckerDelta(UP, DOWN) == 0
    assert KroneckerDelta(DOWN, UP) == 0

    sigma = symbols('sigma')
    for i in [UP, DOWN]:
        assert KroneckerDelta(i, sigma) != 0
        assert KroneckerDelta(sigma, i) != 0


@pytest.fixture(scope='module')
def genmb(spark_ctx):
    """The fixture with a general spin one-half drudge."""
    return SpinOneHalfGenDrudge(spark_ctx)


def test_spin_one_half_general_drudge_has_properties(genmb):
    """Test the basic properties of the drudge."""

    dr = genmb

    assert dr.spin_vals == [UP, DOWN]
    assert dr.orig_ham.n_terms == 2 + 4
    assert dr.ham.n_terms == 2 + 3


def test_restricted_hf_theory(genmb):
    """Test the derivation of restricted HF theory."""

    dr = genmb
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


@pytest.fixture(scope='module')
def parthole(spark_ctx):
    """The fixture with a particle-hole spin one-half drudge."""
    return SpinOneHalfPartHoleDrudge(spark_ctx)


def test_spin_one_half_particle_hole_drudge_has_basic_properties(parthole):
    """Test basic properties of spin one-half particle-hole drudge."""

    dr = parthole
    p = dr.names

    assert dr.orig_ham.n_terms == 8 + 4 * 2 ** 4

    ham_terms = dr.ham.local_terms
    # Numbers are from the old PySLATA code.
    assert len([i for i in ham_terms if len(i.vecs) == 2]) == 8
    assert len([i for i in ham_terms if len(i.vecs) == 4]) == 36
    assert dr.ham.n_terms == 8 + 36


@pytest.fixture(scope='module')
def restricted_parthole(spark_ctx):
    """The fixture with a restricted particle-hole drudge."""
    return RestrictedPartHoleDrudge(spark_ctx)


def test_restricted_parthole_drudge_has_good_hamiltonian(restricted_parthole):
    """Test Hamiltonian of restricted particle-hole drudge.

    Here the original Hamiltonian is going to be compared with the Hamiltonian
    written in terms of the unitary group generators.
    """

    dr = restricted_parthole
    p = dr.names

    e_ = p.e_
    h = dr.one_body
    v = dr.two_body
    half = Rational(1, 2)
    orbs = tuple(dr.orb_ranges)
    p, q, r, s = symbols('p q r s')

    expected_ham = (dr.sum(
        (p, orbs), (q, orbs), h[p, q] * e_[p, q]
    ) + dr.sum(
        (p, orbs), (q, orbs), (r, orbs), (s, orbs),
        half * v[p, r, q, s] * (
            e_[p, q] * e_[r, s] - KroneckerDelta(q, r) * e_[p, s]
        )
    )).simplify()

    assert (dr.orig_ham - expected_ham).simplify() == 0


def test_restricted_parthole_drudge_simplification(restricted_parthole):
    """Test simplification in restricted particle-hole drudge.

    The purpose of this test is mostly on testing the correct resolution of
    ranges of constants up and down.
    """

    dr = restricted_parthole
    p = dr.names
    a = p.a
    sigma = dr.spin_dumms[0]

    op1 = dr.sum(p.c_[a, UP])
    op2 = dr.sum(p.c_dag[a, UP])
    res_concr = (op1 * op2 + op2 * op1).simplify()

    op1 = dr.sum((sigma, dr.spin_range), p.c_[a, sigma])
    res_abstr = (op1 * op2 + op2 * op1).simplify()

    for i in [res_concr, res_abstr]:
        assert (i - Integer(1)).simplify() == 0


def test_restricted_parthole_drudge_on_complex_expression(restricted_parthole):
    """Test simplification entailing complex contraction of spin dummies.

    This tensor comes from an intermediate step in RCCSD theory derivation.
    """

    dr = restricted_parthole
    p = dr.names
    a, b, c, d = p.V_dumms[0:4]
    i, j, k, l = p.O_dumms[0:4]
    e_ = p.e_

    t = IndexedBase('t')
    u = p.u

    tensor = dr.einst(
        u[i, j, c, d] * e_[i, c] * e_[j, d] *
        t[a, b, k, l] * e_[b, l] * e_[a, k]
    )
    res = tensor.simplify()
    frees_vars = res.free_vars
    spin_dumms = set(dr.spin_dumms)
    # The spin dummies are always summed.
    assert not any(i in spin_dumms for i in frees_vars)
