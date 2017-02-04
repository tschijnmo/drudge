"""Tests for the general many-body problem."""

import pytest
from sympy import IndexedBase

from drudge import GenMBDrudge, CR, AN


@pytest.fixture(scope='module')
def genmb(spark_ctx):
    """Initialize the environment for a free algebra."""

    dr = GenMBDrudge(spark_ctx)
    return dr


def test_genmb_has_basic_properties(genmb):
    """Test the general many-body model has basic properties."""
    dr = genmb

    assert len(dr.orb_ranges) == 1
    assert len(dr.spin_vals) == 0

    assert dr.one_body == dr.names.t == IndexedBase('t')
    assert dr.two_body == dr.names.u == IndexedBase('u')

    # The Hamiltonian should already be simplified for this simple model.
    assert dr.ham.n_terms == 2
    assert dr.ham == dr.orig_ham
    # The details of the Hamiltonian will be tested in other ways.


def test_genmb_derives_spin_orbit_hartree_fock(genmb):
    """Test general many-body model can derive HF theory in spin-orbital basis.
    """

    dr = genmb
    p = genmb.names
    c = p.c
    r = p.L
    a, b = p.L_dumms[:2]

    rot = c[CR, a] * c[AN, b]
    comm = (dr.ham | rot).simplify()
    assert comm.n_terms == 4

    rho = IndexedBase('rho')
    # Following Ring and Schuck, here all creation comes before the
    # annihilation.
    res = dr.eval_vev(comm, lambda op1, op2: (
        rho[op2.indices[1], op1.indices[1]]
        if op1.indices[0] == CR and op2.indices[0] == AN
        else 0
    )).simplify()
    assert res.n_terms == 2
    # TODO: Add test of the actual values.
