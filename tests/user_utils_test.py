"""Tests for user utility functions."""

from drudge import Vec, sum_, prod_
from drudge.term import parse_terms


def test_sum_prod_utility():
    """Test the summation and product utility."""

    v = Vec('v')
    vecs = [v[i] for i in range(3)]
    v0, v1, v2 = vecs

    # The proxy object cannot be directly compared.
    assert parse_terms(sum_(vecs)) == parse_terms(v0 + v1 + v2)
    assert parse_terms(prod_(vecs)) == parse_terms(v0 * v1 * v2)

    assert sum_([]) == 0
    assert prod_([]) == 1
