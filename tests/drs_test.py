"""Tests for drudge scripts."""

from sympy import Symbol, IndexedBase, Rational, Integer

from drudge.drs import DrsSymbol, compile_drs
from drudge.utils import sympy_key


#
# Unit tests for the utility classes and functions
# ------------------------------------------------
#


def test_basic_drs_symb():
    """Test the symbol class for basic operations.
    """

    name = 'a'
    ref = Symbol(name)
    dict_ = {ref: 1}

    symbs = [
        DrsSymbol(None, name),
        DrsSymbol([], name)
    ]
    for i in symbs:
        assert isinstance(i, DrsSymbol)
        assert ref == i
        assert i == ref
        assert hash(ref) == hash(i)
        assert dict_[i] == 1
        assert sympy_key(ref) == sympy_key(i)

    ref = Symbol(name + 'x')
    for i in symbs:
        assert ref != i
        assert i != ref
        assert hash(ref) != hash(i)
        assert sympy_key(ref) != sympy_key(i)


def test_basic_drs_indexed():
    """Test basic properties of drudge script indexed object."""

    base_name = 'a'
    orig_base = IndexedBase(base_name)

    for drudge in [None, []]:
        matching_indices = [
            (Symbol('x'), DrsSymbol(drudge, 'x')),
            (
                (Symbol('x'), Symbol('y')),
                (DrsSymbol(drudge, 'x'), DrsSymbol(drudge, 'y'))
            )
        ]
        drs_base = DrsSymbol(drudge, base_name)
        for orig_indices, drs_indices in matching_indices:
            ref = orig_base[orig_indices]
            for i in [
                orig_base[drs_indices],
                drs_base[orig_indices],
                drs_base[drs_indices]
            ]:
                assert ref == i
                assert hash(ref) == hash(i)
                assert sympy_key(ref) == sympy_key(i)


def test_drs_integers():
    """Test fixers for integer literals in drudge scripts."""
    body = 'a = 1 / (1 + (1 + 2))'
    code = compile_drs(body, '<unknown>')
    ctx = {'Integer': Integer}
    exec(code, ctx)
    assert ctx['a'] == Rational(1, 4)
