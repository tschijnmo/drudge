"""Test for the reduced BCS Hamiltonian."""

import pytest
from sympy import KroneckerDelta

from drudge import ReducedBCSDrudge


@pytest.fixture(scope='module')
def rbcs(spark_ctx):
    """Initialize the environment for a reduced BCS problem."""
    return ReducedBCSDrudge(spark_ctx)


def test_rbcs_has_basic_commutations(rbcs: ReducedBCSDrudge):
    """Test the basic commutation rules for Reduced BCS problem."""
    dr = rbcs
    p = dr.names

    # Test access of the basic operators.
    n_, pdag_, p_ = rbcs.cartan, rbcs.raise_, rbcs.lower
    assert n_ is p.N
    assert pdag_ is p.Pdag
    assert p_ is p.P

    # Test commutation without subscripts.
    comm = dr.simplify(n_ | pdag_)
    assert comm == 2 * pdag_
    comm = dr.simplify(n_ | p_)
    assert comm == -2 * p_
    comm = dr.simplify(p_ | pdag_)
    assert comm == 1 - n_

    # Test commutation on the same site.
    i_ = p.i
    n_i = dr.sum(n_[i_])
    pdag_i = dr.sum(pdag_[i_])
    p_i = dr.sum(p_[i_])

    comm = (n_i | pdag_i).simplify()
    assert comm == dr.sum(2 * pdag_i)
    comm = (n_i | p_i).simplify()
    assert comm == dr.sum(-2 * p_i)
    comm = (p_i | pdag_i).simplify()
    assert comm == dr.sum(1 - n_i)

    # Test commutation on different ranges.  This ensures that the delta
    # simplifier is working properly.
    a_ = p.a
    n_a = dr.sum(n_[a_])
    pdag_a = dr.sum(pdag_[a_])
    p_a = dr.sum(p_[a_])

    comm = (n_i | pdag_a).simplify()
    assert comm == 0
    comm = (n_i | p_a).simplify()
    assert comm == 0
    comm = (p_i | pdag_a).simplify()
    assert comm == 0
    comm = (p_i | n_a).simplify()
    assert comm == 0


def test_rbcs_has_basic_commutations_in_fermi(rbcs: ReducedBCSDrudge):
    """Test the pairing commutation rules in terms of fermion operators.

    This function primarily tests the internal function of the reduced BCS
    Hamiltonian.
    """
    dr = rbcs
    p = dr.names

    # Here we are only interested in the same site, since different sites are so
    # easily commutative.

    # Test commutation on the same site.
    a_ = p.a
    n_a = dr.sum(dr.cartan[a_])
    pdag_a = dr.sum(dr.raise_[a_])
    p_a = dr.sum(dr.lower[a_])

    comm = n_a | pdag_a
    assert comm.n_terms == 2
    diff = dr._transl2fermi(comm - dr.sum(2 * pdag_a))
    assert diff.simplify() == 0

    comm = n_a | p_a
    diff = dr._transl2fermi(comm - dr.sum(-2 * p_a))
    assert diff.simplify() == 0

    comm = p_a | pdag_a
    diff = dr._transl2fermi(comm - dr.sum(1 - n_a))
    assert diff.simplify() == 0


def test_rbcs_gives_vev(rbcs: ReducedBCSDrudge):
    """Test VEV utility for reduced BCS problem."""

    dr = rbcs
    p = dr.names

    n_, pdag_, p_ = rbcs.cartan, rbcs.raise_, rbcs.lower
    i_ = p.i
    j_ = p.j
    a_ = p.a

    res = dr.eval_vev(dr.sum(n_[i_]))
    assert res.simplify() == 2

    res = dr.eval_vev(dr.sum(n_[a_]))
    assert res.simplify() == 0

    # Test tensor methods.
    res = dr.sum(pdag_[j_] * p_[i_]).eval_vev()
    assert res == dr.sum(KroneckerDelta(j_, i_).simplify())


def test_rbcs_special_simplification(rbcs: ReducedBCSDrudge):
    """Test the special simplification facilities for pairing algebra."""

    dr = rbcs
    p = dr.names

    n_, pdag_, p_ = rbcs.cartan, rbcs.raise_, rbcs.lower
    a = p.a
    b = p.b

    # Test the simplification based on Cartan with or without index.
    assert dr.simplify(pdag_ * n_) == 0
    assert dr.simplify(pdag_[a] * n_[a]) == 0
    assert dr.simplify(n_ * p_) == 0
    assert dr.simplify(n_[a] * p_[a]) == 0

    # Test the term not conforming with the pattern is not touched.
    assert dr.simplify(pdag_[a] * n_[b]).n_terms == 1
    assert dr.simplify(n_[a] * p_[b]).n_terms == 1


def test_rbcs_has_ham(rbcs: ReducedBCSDrudge):
    """Test the Hamiltonian for the reduced BCS problem."""

    dr = rbcs
    ham = dr.ham
    # Here we tentatively just test the number of terms.
    assert ham.n_terms == 4 + 2
