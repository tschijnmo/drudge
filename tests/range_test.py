"""Tests for the range class."""

from sympy import sympify

from drudge import Range


def test_range_has_basic_operations():
    """Test the basic operations on ranges."""

    a_symb = sympify('a')
    b_symb = sympify('b')

    bound0 = Range('B', 'a', 'b')
    bound1 = Range('B', a_symb, b_symb)
    symb0 = Range('S')
    symb1 = Range('S')

    assert bound0 == bound1
    assert hash(bound0) == hash(bound1)
    assert symb0 == symb1
    assert hash(symb0) == hash(symb1)

    assert bound0 != symb0
    assert hash(bound0) != hash(symb0)

    assert bound0.label == 'B'
    assert bound0.lower == a_symb
    assert bound0.upper == b_symb
    assert bound0.args == (bound1.label, bound1.lower, bound1.upper)
    assert bound0.size == b_symb - a_symb
    assert bound0.replace_label('B1') == Range('B1', a_symb, b_symb)

    assert symb0.label == 'S'
    assert symb0.lower is None
    assert symb0.upper is None
    assert len(symb0.args) == 1
    assert symb0.args[0] == symb1.label


def test_bounds_operations_for_ranges():
    """Properties about bounds for ranges.
    """

    l, u = sympify('l, u')
    r = Range('R')
    assert not r.bounded

    magic = 10
    d = {r: magic}
    assert d[r] == magic

    assert Range('S') not in d

    # Add bounds should not change the identity of the range.
    r_lu = r[l, u]
    assert r_lu.bounded
    assert r_lu.lower == l
    assert r_lu.upper == u
    assert d[r_lu] == 10

    for r_02 in [
        r_lu[0, 2],
        r_lu.map(lambda x: x.xreplace({l: 0, u: 2}))
    ]:
        assert d[r_02] == 10
        assert r_02.bounded
        assert r_02.lower == 0
        assert r_02.upper == 2
        assert r_02.size == 2
