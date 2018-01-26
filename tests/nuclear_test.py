"""Tests for special utilities related to nuclear problems."""

import pytest
from sympy import Symbol, simplify, latex

from drudge import NuclearBogoliubovDrudge
from drudge.nuclear import JOf, TildeOf, MOf, NOf, LOf, PiOf


@pytest.fixture(scope='module')
def nuclear(spark_ctx):
    """Set up the drudge to test."""
    return NuclearBogoliubovDrudge(spark_ctx)


def test_qn_accessors():
    """Test the symbolic functions for quantum number access."""

    k = Symbol('k')
    for acc in [JOf, TildeOf, MOf, NOf, LOf, PiOf]:
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
