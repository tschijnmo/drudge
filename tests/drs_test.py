"""Tests for drudge scripts."""

import types
from unittest.mock import Mock

import pytest
from sympy import Symbol, IndexedBase, Rational, Integer

from drudge import Drudge, Range, Vec, Term
from drudge.drs import (
    DrsSymbol, compile_drs, _DEF_METH_NAME, DrsEnv, _DRUDGE_MAGIC, main
)
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
        assert not hasattr(i, '__dict__')

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


def test_drs_symb_call(spark_ctx):
    """Test calling methods by drs symbols."""

    class TestCls:
        def meth(self):
            return 'meth'

        @property
        def prop(self):
            return 'prop'

    obj = TestCls()
    meth = DrsSymbol(None, 'meth')
    assert meth(obj) == 'meth'
    prop = DrsSymbol(None, 'prop')
    assert prop(obj) == 'prop'
    invalid = DrsSymbol(None, 'invalid')
    with pytest.raises(NameError):
        invalid(obj)
    with pytest.raises(AttributeError) as exc:
        prop.lhs
    assert exc.value.args[0].find('prop') > 0

    # Test automatic raising to tensors.
    v = Vec('v')
    tensor_meth = 'local_terms'
    assert not hasattr(v, tensor_meth)  # Or the test just will not work.
    assert DrsSymbol(Drudge(spark_ctx), tensor_meth)(v) == [
        Term(sums=(), amp=Integer(1), vecs=(v,))
    ]


def test_drs_tensor_def_dispatch(spark_ctx):
    """Tests the dispatch to drudge for tensor definitions."""

    dr = Drudge(spark_ctx)
    names = dr.names

    i_symb = Symbol('i')
    x = IndexedBase('x')
    rhs = x[i_symb]

    dr.add_default_resolver(Range('R'))

    a = DrsSymbol(dr, 'a')
    i = DrsSymbol(dr, 'i')
    for lhs in [a, a[i]]:
        expected = dr.define(lhs, rhs)

        def_ = lhs <= rhs
        assert def_ == expected
        assert not hasattr(names, 'a')
        assert not hasattr(names, '_a')

        def_ = lhs.def_as(rhs)
        assert def_ == expected
        assert names.a == expected
        if isinstance(lhs, DrsSymbol):
            assert names._a == Symbol('a')
        else:
            assert names._a == IndexedBase('a')
        dr.unset_name(def_)


def test_drs_integers():
    """Test fixers for integer literals in drudge scripts."""
    body = 'a = 1 / (1 + (1 + 2))'
    code = compile_drs(body, '<unknown>')
    ctx = {'Integer': Integer}
    exec(code, ctx)
    assert ctx['a'] == Rational(1, 4)


def test_drs_global_def():
    """Test global definition operation in drudge scripts."""
    body = 'a[0] <<= "x"'
    code = compile_drs(body, '<unknown>')
    a = [Mock()]
    def_mock = Mock(return_value=10)
    setattr(a[0], _DEF_METH_NAME, def_mock)
    ctx = {'a': a, 'Integer': Integer}
    exec(code, ctx)

    # Test a is no longer rebound.
    assert ctx['a'] is a
    def_mock.assert_called_with('x')


def test_drs_env():
    """Test the drudge script execution environment."""

    dr = types.SimpleNamespace()
    dr.names = types.SimpleNamespace()
    dr.names.archived = 'archived'

    specials = types.SimpleNamespace()
    specials.special = 'special'

    env = DrsEnv(dr, specials=specials)

    with pytest.raises(KeyError):
        env['__tracebackhide__']

    assert env['archived'] == 'archived'
    assert env['special'] == 'special'
    assert env['names'] is dr.names
    assert env['Range'] is Range
    assert env['Symbol'] is Symbol
    assert env['range'] is range

    # Specially excluded items from some path entries.
    assert isinstance(env['N'], DrsSymbol)


CONF_SCRIPT = """
from dummy_spark import SparkContext
from drudge import Drudge

{} = Drudge(SparkContext())
""".format(_DRUDGE_MAGIC)

DRUDGE_SCRIPT = """
def_ = x <= 1 / 5
"""


def test_drs_main(tmpdir):
    """Test drudge main function."""
    olddir = tmpdir.chdir()
    with open('conf.py', 'w') as fp:
        fp.write(CONF_SCRIPT)
    with open('run.drs', 'w') as fp:
        fp.write(DRUDGE_SCRIPT)

    env = main(['conf.py', 'run.drs'])
    assert 'def_' in env
    def_ = env['def_']
    assert def_.n_terms == 1
    assert def_.rhs_terms[0].amp == Rational(1, 5)

    olddir.chdir()
