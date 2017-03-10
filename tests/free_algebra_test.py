"""Tests for the basic tensor facilities using free algebra."""

import os
import os.path

import pytest
from sympy import (
    sympify, IndexedBase, sin, cos, KroneckerDelta, symbols, conjugate
)

from drudge import Drudge, Range, Vec, Term, Perm, NEG, CONJ


@pytest.fixture(scope='module')
def free_alg(spark_ctx):
    """Initialize the environment for a free algebra."""

    dr = Drudge(spark_ctx)

    r = Range('R')
    dumms = sympify('i, j, k, l, m, n')
    dr.set_dumms(r, dumms)

    s = Range('S')
    s_dumms = symbols('alpha beta')
    dr.set_dumms(s, s_dumms)

    dr.add_resolver_for_dumms()

    v = Vec('v')
    dr.set_name(v)

    m = IndexedBase('m')
    dr.set_symm(m, Perm([1, 0], NEG))

    h = IndexedBase('h')
    dr.set_symm(h, Perm([1, 0], NEG | CONJ))

    dr.set_tensor_method('get_one', lambda x: 1)

    return dr


def test_drudge_has_names(free_alg):
    """Test the name archive for drudge objects.

    Here selected names are tested to makes sure all the code are covered.
    """

    p = free_alg.names

    # Range and dummy related.
    assert p.R == Range('R')
    assert len(p.R_dumms) == 6
    assert p.R_dumms[0] == p.i
    assert p.R_dumms[-1] == p.n

    # Vector bases.
    assert p.v == Vec('v')

    # Scalar bases.
    assert p.m == IndexedBase('m')


def test_tensor_can_be_created(free_alg):
    """Test simple tensor creation."""

    dr = free_alg
    p = dr.names
    i, v, r = p.i, p.v, p.R
    x = IndexedBase('x')

    # Create the tensor by two user creation functions.
    for tensor in [
        dr.sum((i, r), x[i] * v[i]),
        dr.einst(x[i] * v[i])
    ]:
        assert tensor.n_terms == 1

        terms = tensor.local_terms
        assert len(terms) == 1
        term = terms[0]
        assert term == Term(((i, r),), x[i], (v[i],))


def test_tensor_has_basic_operations(free_alg):
    """Test some of the basic operations on tensors.

    Tested in this module:

        1. Addition.
        2. Merge.
        3. Free variable.
        4. Dummy reset.
        5. Equality comparison.
        6. Expansion
        7. Mapping to scalars.
        8. Base presence testing.
    """

    dr = free_alg
    p = dr.names
    i, j, k, l, m = p.R_dumms[:5]
    x = IndexedBase('x')
    r = p.R
    v = p.v
    tensor = (
        dr.sum((l, r), x[i, l] * v[l]) +
        dr.sum((m, r), x[j, m] * v[m])
    )

    # Without dummy resetting, they cannot be merged.
    assert tensor.n_terms == 2
    assert tensor.merge().n_terms == 2

    # Free variables are important for dummy resetting.
    free_vars = tensor.free_vars
    assert free_vars == {x.label, i, j}

    # Reset dummy.
    reset = tensor.reset_dumms()
    expected = (
        dr.sum((k, r), x[i, k] * v[k]) +
        dr.sum((k, r), x[j, k] * v[k])
    )
    assert reset == expected
    assert reset.local_terms == expected.local_terms

    # Merge the terms.
    merged = reset.merge()
    assert merged.n_terms == 1
    term = merged.local_terms[0]
    assert term == Term(((k, r),), x[i, k] + x[j, k], (v[k],))

    # Slightly separate test for expansion.
    c, d = symbols('c d')
    tensor = dr.sum((i, r), x[i] * (c + d) * v[i])
    assert tensor.n_terms == 1
    expanded = tensor.expand()
    assert expanded.n_terms == 2

    # Here we also test concrete summation facility.
    expected = dr.sum(
        (i, r), (j, [c, d]), x[i] * j * v[i]
    )
    assert set(expected.local_terms) == set(expected.local_terms)

    # Test mapping to scalars.
    tensor = dr.sum((i, r), x[i] * v[i, j])
    y = IndexedBase('y')
    substs = {x: y, j: c}
    res = tensor.map2scalars(lambda x: x.xreplace(substs))
    assert res == dr.sum((i, r), y[i] * v[i, c])
    res = tensor.map2scalars(lambda x: x.xreplace(substs), skip_vecs=True)
    assert res == dr.sum((i, r), y[i] * v[i, j])

    # Test base presence.
    tensor = dr.einst(x[i] * v[i])
    assert tensor.has_base(x)
    assert tensor.has_base(v)
    assert not tensor.has_base(IndexedBase('y'))
    assert not tensor.has_base(Vec('w'))


def test_tensor_can_be_simplified_amp(free_alg):
    """Test the amplitude simplification for tensors.

    More than trivial tensor amplitude simplification is tested here.  Currently
    it mostly concentrates on the dispatching to SymPy and delta simplification.
    The master simplification is also tested.
    """

    dr = free_alg
    p = dr.names
    r = p.R
    s = p.S
    v = p.v
    i, j = p.R_dumms[:2]
    alpha = p.alpha

    x = IndexedBase('x')
    y = IndexedBase('y')
    theta = sympify('theta')

    tensor = (
        dr.sum((i, r), sin(theta) ** 2 * x[i] * v[i]) +
        dr.sum(
            (i, r), (j, r),
            cos(theta) ** 2 * x[j] * KroneckerDelta(i, j) * v[i]
        ) +
        dr.sum((i, r), (alpha, s), KroneckerDelta(i, alpha) * y[i] * v[i])
    )
    assert tensor.n_terms == 3

    first = tensor.simplify_deltas().simplify_amps()
    # Now we should have one term killed.
    assert first.n_terms == 2

    # Merge again should really simplify.
    merged = first.reset_dumms().merge().simplify_amps()
    assert merged.n_terms == 1
    expected = dr.sum((i, r), x[i] * v[i])
    assert merged == expected

    # The master simplification should do it in one turn.
    simpl = tensor.simplify()
    assert simpl == expected


def test_tensor_can_be_canonicalized(free_alg):
    """Test tensor canonicalization in simplification.

    The master simplification function is tested, the core simplification is at
    the canonicalization.  Equality testing with zero is also tested.
    """

    dr = free_alg
    p = dr.names
    i, j = p.R_dumms[:2]
    r = p.R
    m = p.m
    h = p.h
    v = p.v

    # Anti-symmetric real matrix.
    tensor = (
        dr.sum((i, r), (j, r), m[i, j] * v[i] * v[j]) +
        dr.sum((i, r), (j, r), m[j, i] * v[i] * v[j])
    )
    assert tensor.n_terms == 2

    res = tensor.simplify()
    assert res == 0

    # Hermitian matrix.
    tensor = dr.einst(
        h[i, j] * v[i] * v[j] + conjugate(h[j, i]) * v[i] * v[j]
    )
    assert tensor.n_terms == 2
    res = tensor.simplify()
    assert res == 0


def test_tensor_math_ops(free_alg):
    """Test tensor math operations.

    Mainly here we test addition and multiplication.
    """

    dr = free_alg
    p = dr.names
    r = p.R
    v = p.v
    w = Vec('w')
    x = IndexedBase('x')
    i, j, k = p.R_dumms[:3]
    a = sympify('a')

    v1 = dr.sum((i, r), x[i] * v[i])
    w1 = dr.sum((i, r), x[i] * w[i])
    assert v1.n_terms == 1
    assert w1.n_terms == 1

    v1_1 = v1 + 2
    assert v1_1.n_terms == 2
    assert v1_1 == 2 + v1

    w1_1 = w1 + a
    assert w1_1.n_terms == 2
    assert w1_1 == a + w1

    prod = v1_1 * w1_1
    # Test scalar multiplication here as well.
    expected = (
        2 * a + a * v1 + 2 * w1 +
        dr.sum((i, r), (j, r), x[i] * x[j] * v[i] * w[j])
    )
    assert prod.simplify() == expected.simplify()

    # Test the commutator operation.
    comm_v1v1 = v1 | v1
    assert comm_v1v1.simplify() == 0
    # Here the tensor subtraction can also be tested.
    comm_v1w1 = v1 | w1
    expected = (
        dr.sum((i, r), (j, r), x[i] * x[j] * v[i] * w[j]) -
        dr.sum((i, r), (j, r), x[j] * x[i] * w[i] * v[j])
    )
    assert comm_v1w1.simplify() == expected.simplify()


def test_tensors_can_be_differentiated(free_alg):
    """Test the analytic gradient computation of tensors."""

    dr = free_alg
    p = dr.names

    a = IndexedBase('a')
    b = IndexedBase('b')
    i, j, k, l, m, n = p.R_dumms[:6]

    tensor = dr.einst(
        a[i, j, k, l] * b[i, j] * conjugate(b[k, l])
    )

    # Test real analytic gradient.

    res = tensor.diff(b[i, j], real=True)
    expected = dr.einst(
        b[k, l] * (a[k, l, i, j] + a[i, j, k, l])
    )
    assert (res - expected).simplify() == 0

    # Test Wirtinger complex derivative.
    res, res_conj = [
        tensor.diff(b[m, n], wirtinger_conj=conj)
        for conj in [False, True]
        ]

    expected = dr.einst(
        conjugate(b[i, j]) * a[m, n, i, j]
    )
    expect_conj = dr.einst(
        a[i, j, m, n] * b[i, j]
    )

    for res_i, expected_i in [(res, expected), (res_conj, expect_conj)]:
        assert (res_i - expected_i).simplify() == 0

    # Test real analytic gradient with a simple test case.

    tensor = dr.einst(b[i, j] * b[j, i])
    grad = tensor.diff(b[i, j])
    assert (grad - 2 * b[j, i]).simplify() == 0


def test_tensors_can_be_substituted_scalars(free_alg):
    """Test scalar substitution facility for tensors."""

    dr = free_alg
    p = dr.names

    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    r = p.R
    i, j, k, l, m = p.R_dumms[:5]

    x_def = dr.define(
        x[i], dr.sum((j, r), y[j] * z[i])
    )
    orig = dr.sum((i, r), x[i] ** 2 * x[k])

    # k is free.
    expected = dr.sum(
        (i, r), (j, r), (l, r), (m, r),
        z[i] ** 2 * y[j] * y[l] * y[m] * z[k]
    )

    # Test different ways to perform the substitution.
    for res in [
        orig.subst(x[i], x_def.rhs),
        orig.subst_all([x_def]),
        orig.subst_all([(x[i], x_def.rhs)]),
        x_def.act(orig)
    ]:
        assert res.simplify() == expected.simplify()


def test_tensors_can_be_substituted_vectors(free_alg):
    """Test vector substitution facility for tensors."""

    dr = free_alg
    p = dr.names

    x = IndexedBase('x')
    t = IndexedBase('t')
    u = IndexedBase('u')
    i, j = p.i, p.j
    v = p.v
    w = Vec('w')

    orig = dr.einst(x[i] * v[i])
    v_def = dr.einst(t[i, j] * w[j] + u[i, j] * w[j])

    res = orig.subst(v[i], v_def).simplify()

    expected = dr.einst(
        x[i] * t[i, j] * w[j] + x[i] * u[i, j] * w[j]
    ).simplify()
    assert res == expected


def test_tensor_method(free_alg):
    """Test tensor method can be injected."""

    tensor = free_alg.sum(10)
    assert tensor.get_one() == 1

    with pytest.raises(AttributeError):
        tensor.get_two()


def test_tensor_def_creation_and_basic_properties(free_alg):
    """Test basic tensor definition creation and properties.

    Since tensor definitions are more frequently used for scalars, here we
    concentrate more on the scalar quantities than on vectors.
    """

    dr = free_alg
    p = dr.names
    i, j, k = p.R_dumms[:3]

    x = IndexedBase('x')
    o = IndexedBase('o')
    y = IndexedBase('y')

    y_def = dr.define(y, (i, p.R), dr.sum((j, p.R), o[i, j] * x[j]))

    assert y_def.is_scalar
    assert y_def.rhs == dr.einst(o[i, j] * x[j])
    assert y_def.lhs == y[i]
    assert y_def.base == y
    assert y_def.exts == [(i, p.R)]

    assert str(y_def) == 'y[i] = sum_{j} x[j]*o[i, j]'
    assert y_def.latex().strip() == r'y_{i} = \sum_{j \in R} x_{j} o_{i,j}'

    y_def1 = dr.define(y[i], dr.sum((j, p.R), o[i, j] * x[j]))
    y_def2 = dr.define_einst(y[i], o[i, j] * x[j])
    assert y_def1 == y_def
    assert y_def2 == y_def

    # This tests the `act` method as well.
    assert y_def[1].simplify() == dr.einst(o[1, j] * x[j]).simplify()


def test_tensors_has_string_and_latex_form(free_alg, tmpdir):
    """Test the string and LaTeX form representation of tensors."""

    dr = free_alg
    p = dr.names

    v = p.v
    i = p.i
    x = IndexedBase('x')

    tensor = dr.einst(x[i] * v[i] - x[i] * v[i])
    zero = tensor.simplify()

    # The basic string form.
    orig = str(tensor)
    assert orig == 'sum_{i} x[i] * v[i]\n + sum_{i} -x[i] * v[i]'
    assert str(zero) == '0'

    # The LaTeX form.
    expected = (
        r'\sum_{i \in R} x_{i} \mathbf{v}_{i} '
        '- \sum_{i \in R} x_{i} \mathbf{v}_{i}'
    )
    assert tensor.latex() == expected
    assert tensor.latex(sep_lines=True) != expected
    assert tensor.latex(sep_lines=True).replace(r'\\ ', '') == expected
    assert zero.latex() == '0'
    assert zero.latex(sep_lines=True) == '0'

    # Test the reporting facility.
    with tmpdir.as_cwd():
        filename = 'freealg.html'
        with dr.report(filename, 'Simple report test') as rep:
            rep.add('A simple tensor', tensor, description='Nothing')

        # Here we just simply test the existence of the file.
        assert os.path.isfile(filename)
        os.remove(filename)


def test_drudge_has_default_properties(free_alg):
    """Test some basic default properties for drudge objects."""

    assert isinstance(free_alg.num_partitions, int)
    assert free_alg.full_simplify
    assert not free_alg.simple_merge


def test_tensor_can_be_added_summation(free_alg):
    """Test addition of new summations for existing tensors."""

    dr = free_alg
    p = dr.names
    i, j = p.R_dumms[:2]
    x = IndexedBase('x')
    y = IndexedBase('y')

    tensor = dr.sum((i, p.R), x[i, j] * y[j, i])

    for res in [
        dr.einst(tensor),
        dr.sum((j, p.R), tensor)
    ]:
        assert res == dr.einst(x[i, j] * y[j, i])
