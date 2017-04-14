"""Tests on the particle-hole model."""

import pytest
from sympy import Rational, IndexedBase

from drudge import PartHoleDrudge


@pytest.fixture(scope='module')
def parthole(spark_ctx):
    """Initialize the environment for a free algebra."""
    dr = PartHoleDrudge(spark_ctx)
    return dr


@pytest.mark.parametrize('par_level', [0, 1, 2])
@pytest.mark.parametrize('full_simplify', [True, False])
@pytest.mark.parametrize('simple_merge', [True, False])
def test_simple_parthole_normal_order(
        parthole, par_level, full_simplify, simple_merge
):
    """Test particle-hole normal ordering on a simple term.

    Here we just have a term normal-ordered in terms of bare electrons but it
    not normal ordered in terms of quasi-partitions.  This makes sure that the
    model correctly treats the problem.
    """

    dr = parthole
    p = dr.names
    c_dag = p.c_dag
    c_ = p.c_
    i = p.i
    j = p.j

    t = dr.one_body
    inp = dr.einst(
        t[i, j] * c_dag[i] * c_[j]
    )

    dr.wick_parallel = par_level
    dr.full_simplify = full_simplify
    dr.simple_merge = simple_merge

    res = inp.simplify()

    dr.wick_parallel = 0
    dr.full_simplify = True
    dr.simple_merge = False

    assert res.n_terms == 2
    assert res == dr.einst(
        -t[i, j] * c_[j] * c_dag[i] + t[i, i]
    ).simplify()


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


def test_parthole_drudge_gives_conventional_dummies(parthole):
    """Test dummy naming in canonicalization facility on particle-hole drudge.
    """

    dr = parthole
    p = dr.names
    c_dag = p.c_dag
    c_ = p.c_
    a, b = p.a, p.b
    i, j = p.i, p.j
    u = p.u

    tensor = dr.einst(u[a, b, i, j] * c_dag[a] * c_dag[b] * c_[j] * c_[i])
    res = tensor.simplify()
    assert res == tensor
