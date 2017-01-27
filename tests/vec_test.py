"""Tests for vectors."""

from sympy import sympify

from drudge import Vec


def test_vecs_has_basic_properties():
    """Tests the basic properties of vector instances."""

    base = Vec('v')
    v_ab = Vec('v', indices=['a', 'b'])
    v_ab_1 = base['a', 'b']
    v_ab_2 = (base['a'])['b']

    indices_ref = (sympify('a'), sympify('b'))
    hash_ref = hash(v_ab)
    str_ref = 'v[a, b]'
    repr_ref = "Vec('v', (a, b))"

    for i in [v_ab, v_ab_1, v_ab_2]:
        assert i.label == base.label
        assert i.base == base
        assert i.indices == indices_ref
        assert hash(i) == hash_ref
        assert i == v_ab
        assert str(i) == str_ref
        assert repr(i) == repr_ref
