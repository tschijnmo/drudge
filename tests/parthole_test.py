"""Tests on the particle-hole model."""

import pytest
from sympy import Rational, IndexedBase

from drudge import PartHoleDrudge, CR, AN
from drudge.wick import wick_expand_term


@pytest.fixture(scope='module')
def parthole(spark_ctx):
    """Initialize the environment for a free algebra."""
    dr = PartHoleDrudge(spark_ctx)
    return dr


def test_parthole_normal_order_on_term(parthole):
    """Test particle-hole normal ordering on a simple term.

    This test act on a tensor term directly without parallelization.  It is
    supposed for the ease of debugging.
    """

    dr = parthole
    p = dr.names
    c_ = dr.op
    i = p.i
    j = p.j

    t = dr.one_body
    term = dr.sum(
        (i, p.O), (j, p.O), t[i, j] * c_[CR, i] * c_[AN, j]
    ).local_terms[0]

    res = wick_expand_term(
        term, comparator=dr.comparator, contractor=dr.contractor,
        phase=dr.phase, symms=dr.symms.value
    )

    # Bare minimum inspection.
    assert len(res) == 2


def test_parthole_drudge_has_good_ham(parthole):
    """Test the Hamiltonian of the particle-hole model."""

    dr = parthole
    p = dr.names

    # Minimum inspection.
    #
    # TODO: Add inspection of the actual value.

    assert dr.orig_ham.n_terms == 2 ** 2 + 2 ** 4
    assert dr.full_ham.n_terms == 2 + 8 + 9

    assert dr.ham_energy.n_terms == 2
    assert dr.one_body_ham.n_terms == 8
    assert dr.ham.n_terms == 4 + 9

    # Here we test the simplest vacuum energy.  In spite of its simplicity, it
    # should cover a large part of the code.

    h_range = p.O
    i, j = p.O_dumms[:2]
    expected = (dr.sum(
        (i, h_range), dr.one_body[i, i]
    ) + dr.sum(
        (i, h_range), (j, h_range), dr.two_body[i, j, i, j] * Rational(1, 2)
    )).simplify()

    assert dr.eval_fermi_vev(dr.orig_ham).simplify() == expected
    assert dr.ham_energy == expected


def test_tce_parse(parthole):
    """Test the parsing of TCE output.

    This test just tests one line in the CCD amplitude equation that contains
    most of the features in the TCE output files.
    """

    dr = parthole

    tce_out = """
    [ - 1.0 + 1.0 * P( p3 p4 h1 h2 => p3 p4 h2 h1 ) ] \
    * Sum ( h5 ) * f ( h5 h1 ) * t ( p3 p4 h5 h2 )
    """

    t = IndexedBase('t')

    res = dr.parse_tce(tce_out, {2: t})

    p = dr.names
    a, b = p.V_dumms[:2]
    i, j, k = p.O_dumms[:3]
    f = dr.fock
    expected = dr.sum(
        (k, p.O), -f[k, i] * t[a, b, k, j] + f[k, j] * t[a, b, k, i]
    )

    assert res.simplify() == expected.simplify()


def test_parthole_with_ph_excitations(parthole):
    """Test the capability of particle-hole drudge by excitations.

    The drudge should be able to find that the particle-hole excitations
    commutes with each other.
    """

    dr = parthole
    p = dr.names
    a, b = p.V_dumms[:2]
    i, j = p.O_dumms[:2]
    c_ = p.c_
    c_dag = p.c_dag

    # Without summation.
    excit_1 = dr.sum(c_dag[a] * c_[i])
    excit_2 = dr.sum(c_dag[b] * c_[j])
    assert (excit_1 | excit_2).simplify() == 0

    # With summation.
    excit_1 = dr.sum((a, p.V), (i, p.O), c_dag[a] * c_[i])
    excit_2 = dr.sum((a, p.V), (i, p.O), c_dag[a] * c_[i])
    comm = excit_1 | excit_2
    for i in comm.local_terms:
        assert len(i.sums) == 4
    assert comm.simplify() == 0
