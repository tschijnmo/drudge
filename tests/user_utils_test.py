"""Tests for user utility functions."""

import time
import types
from unittest.mock import MagicMock

from drudge import Vec, sum_, prod_, TimeStamper
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


def test_time_stamper():
    """Test the time stamper utility."""

    tensor = types.SimpleNamespace(n_terms=2, cache=MagicMock())

    stamper = TimeStamper()
    time.sleep(0.5)
    res = stamper.stamp('Nothing')
    assert res.startswith('Nothing done')
    assert float(res.split()[-2]) - 0.5 < 0.1

    time.sleep(0.5)
    res = stamper.stamp('Tensor', tensor)
    assert res.startswith('Tensor done, 2 terms')
    assert float(res.split()[-2]) - 0.5 < 0.1
    tensor.cache.assert_called_once_with()
