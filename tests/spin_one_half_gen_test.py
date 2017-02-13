"""Tests for the general model with explicit one-half spin."""

import pytest

from drudge import UP, DOWN, SpinOneHalfGenDrudge


@pytest.fixture(scope='module')
def dr(spark_ctx):
    """The fixture with a general spin one-half drudge."""
    return SpinOneHalfGenDrudge(spark_ctx)


def test_spin_one_half_general_drudge_has_properties(dr):
    """Test the basic properties of the drudge."""

    assert dr.spin_vals == [UP, DOWN]
    assert dr.orig_ham.n_terms == 2 + 4
    assert dr.ham.n_terms == 2 + 3
