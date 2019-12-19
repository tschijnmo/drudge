"""Tests for the general quadratic algebra drudge."""

import pytest
from sympy import Integer
from drudge import Vec, GenQuadLatticeDrudge


def test_assume_comm(spark_ctx):
    """
    Test the case where vectors are assumed to commute if there commutator
    is not specified.
    """

    v1 = Vec(r'v_1')
    v2 = Vec(r'v_2')

    dr = GenQuadLatticeDrudge(spark_ctx, order=(v1, v2), comms={},
                              assume_comm=True)
    tensor = dr.sum(v1 | v2)

    assert tensor.simplify() == Integer(0)
