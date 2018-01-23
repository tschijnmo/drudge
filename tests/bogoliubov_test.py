"""Test of the Bogoliubov facility."""

import itertools
import math

import pytest
from sympy import Indexed, Symbol, IndexedBase, symbols, conjugate

from drudge import BogoliubovDrudge, CR, AN


@pytest.fixture(scope='module')
def bogoliubov(spark_ctx):
    """Initialize the environment for a free algebra."""

    dr = BogoliubovDrudge(spark_ctx)
    return dr


def test_bogoliubov_has_hamiltonian(bogoliubov: BogoliubovDrudge):
    """Test the Hamiltonian in the Bogoliubov problem."""
    dr = bogoliubov
    qp_range = dr.qp_range

    orders = {}
    for term in dr.ham.local_terms:
        cr_indices = []
        an_indices = []
        for i in term.vecs:
            assert i.base == dr.qp_op

            assert len(i.indices) == 2
            char, index = i.indices
            if char == CR:
                cr_indices.append(index)
                assert len(an_indices) == 0
            elif char == AN:
                an_indices.append(index)
            else:
                assert False
            continue

        an_indices.reverse()
        indices = tuple(itertools.chain(cr_indices, an_indices))
        order = (len(cr_indices), len(an_indices))

        assert dict(term.sums) == {
            i: qp_range for i in indices
        }

        # Here we use Python facility to test against the SymPy factorial in the
        # code.
        norm = math.factorial(order[0]) * math.factorial(order[1])

        assert order not in orders
        if order == (0, 0):
            assert isinstance(term.amp, Symbol)
        else:
            amp = (term.amp * norm).simplify()
            assert isinstance(amp, Indexed)
            assert amp.indices == indices
            orders[order] = amp.base

        continue

    # Right now we do not test the bases.
    assert set(orders.keys()) == {
        (1, 1),
        (2, 0),
        (0, 2),
        (4, 0),
        (0, 4),
        (3, 1),
        (1, 3),
        (2, 2)
    }


def test_bogoliubov_has_correct_matrix_elements(bogoliubov: BogoliubovDrudge):
    """Test the correctness of Bogoliubov Matrix elements."""
    dr = bogoliubov
    p_range = dr.p_range
    mes = dr.ham_mes

    # Here we first check simple ones.
    def_40 = None
    def_00 = None
    for i in mes:
        if i.base == IndexedBase('H^{40}'):
            assert def_40 is None
            def_40 = i
        elif i.base == Symbol('H^{00}'):
            assert def_00 is None
            def_00 = i
        continue
    assert def_40 is not None
    assert def_00 is not None

    # Test (4, 0).
    #
    # This test is deficient in that it is only a valid form, the full symmetry
    # for k1...k4 cannot be fully considered here.  So correct result could
    # still potentially fail this test in principle.
    ext_symbs = [i for i, _ in def_40.exts]
    k1, k2, k3, k4 = ext_symbs
    l1, l2, l3, l4 = symbols('l1 l2 l3 l4')
    assert dr.simplify(def_40.rhs - dr.sum(
        (l1, p_range), (l2, p_range), (l3, p_range), (l4, p_range),
        -6 * dr.two_body[l1, l2, l3, l4] * conjugate(dr.u_base[l1, k1])
        * conjugate(dr.u_base[l2, k2])
        * conjugate(dr.v_base[l4, k4])
        * conjugate(dr.v_base[l3, k3])
    )) == 0

    # Test the registration of the matrix element names.
    assert hasattr(dr.names, 'H00')
    assert isinstance(dr.names.H00, IndexedBase)


def test_bogoliubov_vev(bogoliubov: BogoliubovDrudge):
    """Test the correctness of Bogoliubov VEV evaluation."""
    dr = bogoliubov
    res = dr.ham.eval_bogoliubov_vev()
    assert res == dr.sum(Symbol(r'H^{00}'))
