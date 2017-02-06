"""Tests on the particle-hole model."""

import pytest

from drudge import PartHoleDrudge, CR, AN
from drudge.wick import wick_expand


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

    res = wick_expand(
        term, comparator=dr.comparator, contractor=dr.contractor,
        phase=dr.phase, symms=dr.symms.value
    )

    # Bare minimum inspection.
    assert len(res) == 2


def test_parthole_drudge_has_good_ham(parthole):
    """Test the Hamiltonian of the particle-hole model."""

    dr = parthole

    # Minimum inspection.
    #
    # TODO: Add inspection of the actual value.

    assert dr.orig_ham.n_terms == 2 ** 2 + 2 ** 4
    assert dr.full_ham.n_terms == 2 + 8 + 9

    assert dr.ham_energy.n_terms == 2
    assert dr.one_body_ham.n_terms == 8
    assert dr.ham.n_terms == 4 + 9
