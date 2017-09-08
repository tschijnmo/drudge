"""Tests for user utility functions."""

import time
import types
from unittest.mock import MagicMock

from sympy import IndexedBase, symbols

from drudge import Vec, sum_, prod_, Stopwatch, ScalarLatexPrinter
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


def test_stopwatch():
    """Test the stopwatch utility."""

    tensor = types.SimpleNamespace(n_terms=2, cache=MagicMock())
    res_holder = [None]

    def print_cb(stamp):
        res_holder[0] = stamp

    stamper = Stopwatch(print_cb)
    time.sleep(0.5)
    stamper.tock('Nothing')
    res = res_holder[0]
    assert res.startswith('Nothing done')
    assert float(res.split()[-2]) - 0.5 < 0.1

    time.sleep(0.5)
    stamper.tock('Tensor', tensor)
    res = res_holder[0]
    assert res.startswith('Tensor done, 2 terms')
    assert float(res.split()[-2]) - 0.5 < 0.1
    tensor.cache.assert_called_once_with()

    stamper.tock_total()
    res = res_holder[0]
    assert float(res.split()[-2]) - 1.0 < 0.1


def test_scalar_latex_printing():
    """Test the printing of scalars into LaTeX form."""

    x1 = IndexedBase('x1')
    i, j = symbols('i j')
    expr = x1[i, j]
    assert ScalarLatexPrinter().doprint(expr) == 'x^{(1)}_{i,j}'
