"""Tests for the general many-body problem.

Some general tensor operations requiring range and dummies in the drudge are
also tested here.
"""

import pytest
from sympy import IndexedBase, conjugate, Symbol, symbols, I, exp, pi, sqrt

from drudge import GenMBDrudge, CR, AN, Range


@pytest.fixture(scope='module')
def genmb(spark_ctx):
    """Initialize the environment for a free algebra."""

    dr = GenMBDrudge(spark_ctx)
    return dr


def test_genmb_has_basic_properties(genmb):
    """Test the general many-body model has basic properties."""
    dr = genmb

    assert len(dr.orb_ranges) == 1
    assert dr.spin_vals is None

    assert dr.one_body == dr.names.t == IndexedBase('t')
    assert dr.two_body == dr.names.u == IndexedBase('u')

    # The Hamiltonian should already be simplified for this simple model.
    assert dr.ham.n_terms == 2
    assert dr.ham == dr.orig_ham
    # The details of the Hamiltonian will be tested in other ways.


def test_einstein_summation(genmb):
    """Test Einstein summation convention."""
    dr = genmb
    p = dr.names
    l = p.L
    a, b, c = p.L_dumms[:3]
    o = IndexedBase('o')
    v = IndexedBase('v')
    c_dag = p.c_dag

    summand = o[a, b] * v[b] * c_dag[a]
    tensor = dr.einst(summand)
    assert tensor == dr.sum((a, l), (b, l), summand)

    summand = conjugate(o[b, a]) * v[b] * c_dag[a]
    tensor = dr.einst(summand)
    assert tensor == dr.sum((a, l), (b, l), summand)


@pytest.mark.parametrize('par_level', [0, 1, 2])
@pytest.mark.parametrize('full_simplify', [True, False])
@pytest.mark.parametrize('simple_merge', [True, False])
def test_genmb_simplify_simple_expressions(
        genmb, par_level, full_simplify, simple_merge
):
    """Test the basic Wick expansion facility on a single Fermion expression."""

    dr = genmb  # type: GenMBDrudge

    c_ = dr.op[AN]
    c_dag = dr.op[CR]
    r = dr.names.L
    a, b, c, d = dr.names.L_dumms[:4]

    t = IndexedBase('t')
    u = IndexedBase('u')

    inp = dr.sum(
        (a, r), (b, r), (c, r), (d, r),
        t[a, b] * u[c, d] * c_dag[a] * c_[b] * c_dag[c] * c_[d]
    )

    dr.wick_parallel = par_level
    assert dr.wick_parallel == par_level
    dr.full_simplify = full_simplify
    assert dr.full_simplify == full_simplify
    dr.simple_merge = simple_merge
    assert dr.simple_merge == simple_merge

    res = inp.simplify()

    dr.wick_parallel = 0
    assert dr.wick_parallel == 0
    dr.full_simplify = True
    assert dr.full_simplify
    dr.simple_merge = False
    assert not dr.simple_merge

    assert res.n_terms == 2

    expected = dr.einst(
        t[a, c] * u[b, d] * c_dag[a] * c_dag[b] * c_[d] * c_[c] +
        t[a, c] * u[c, b] * c_dag[a] * c_[b]
    ).simplify()

    assert res == expected


def test_genmb_simplifies_nilpotent_operators(genmb):
    """Test simplification of tensors vanishing by nilpotency."""

    dr = genmb
    p = dr.names
    c_dag = p.c_dag
    c_ = p.c_
    a = p.a

    assert dr.sum(c_dag[a] * c_dag[a]).simplify() == 0
    assert dr.sum(c_[a] * c_[a]).simplify() == 0
    assert dr.sum(c_dag[a] * c_[a]).simplify().n_terms == 1


def test_genmb_gives_conventional_dummies(genmb):
    """Test dummy placement in canonicalization facility on many-body drudge.
    """

    dr = genmb
    p = dr.names
    c_dag = p.c_dag
    c_ = p.c_
    a, b, c, d = p.a, p.b, p.c, p.d

    x = IndexedBase('x')
    tensor = dr.einst(x[a, b, c, d] * c_dag[a] * c_dag[b] * c_[d] * c_[c])
    res = tensor.simplify()
    assert res == tensor


def test_genmb_derives_spin_orbit_hartree_fock(genmb):
    """Test general many-body model can derive HF theory in spin-orbital basis.
    """

    dr = genmb
    p = genmb.names
    c_ = p.c_
    c_dag = p.c_dag
    r = p.L
    a, b, c, d = p.L_dumms[:4]

    rot = c_dag[a] * c_[b]
    comm = (dr.ham | rot).simplify()
    assert comm.n_terms == 4

    rho = IndexedBase('rho')
    # Following Ring and Schuck, here all creation comes before the
    # annihilation.
    res = comm.eval_vev(lambda op1, op2, _: (
        rho[op2.indices[1], op1.indices[1]]
        if op1.indices[0] == CR and op2.indices[0] == AN
        else 0
    )).simplify()
    assert res.n_terms == 6

    # The correct result: [\rho, f]^b_a

    f = IndexedBase('f')
    expected = dr.sum((c, r), rho[b, c] * f[c, a] - f[b, c] * rho[c, a])
    expected = expected.subst(f[a, b], p.t[a, b] + dr.sum(
        (c, r), (d, r),
        p.u[a, c, b, d] * rho[d, c] - p.u[a, c, d, b] * rho[d, c]
    ))
    expected = expected.simplify()

    assert res == expected


def test_fock_drudge_prints_operators(genmb):
    """Test the LaTeX printing by Fock drudge.

    Things like term linkage should be tested for the base class.  Here we
    concentrate on the vector part, which is turned for field operators.
    """

    dr = genmb
    p = dr.names

    x = IndexedBase('x')
    a, b = p.L_dumms[:2]

    tensor = dr.einst(- x[a, b] * p.c_dag[a] * p.c_[b])
    assert tensor.latex() == (
        r'- \sum_{a \in L} \sum_{b \in L} x_{a,b}    c^{\dagger}_{a} c^{}_{b}'
    )


def test_dagger_of_field_operators(genmb):
    """Test taking the Hermitian adjoint of field operators."""

    dr = genmb
    p = dr.names
    x = IndexedBase('x')
    c_dag = p.c_dag
    c_ = p.c_
    a, b = p.L_dumms[:2]

    tensor = dr.einst(x[a, b] * c_dag[a] * c_[b])

    real_dag = tensor.dagger(real=True)
    assert real_dag == dr.einst(x[a, b] * c_dag[b] * c_[a])

    compl_dag = tensor.dagger()
    assert compl_dag == dr.einst(conjugate(x[a, b]) * c_dag[b] * c_[a])


def test_diag_tight_binding_hamiltonian(spark_ctx):
    """Test automatic diagonalization of the tight-binding Hamiltonian.

    The primary target of this test is the simplification of amplitude
    summations.
    """

    n = Symbol('N', integer=True)
    dr = GenMBDrudge(spark_ctx, orb=(
        (Range('L', 0, n), symbols('x y z x1 x2', integer=True)),
    ))

    # The reciprocal space range and dummies.
    k, q = symbols('k q', integer=True)
    dr.set_dumms(Range('R', 0, n), [k, q])

    p = dr.names
    h = Symbol('h')  # Hopping neighbours.
    delta = Symbol('Delta')
    c_dag = p.c_dag
    c_ = p.c_
    a = p.L_dumms[0]

    # Hamiltonian in the real-space
    real_ham = dr.sum(
        (a, p.L), (h, 1, -1), delta * c_dag[a + h] * c_[a]
    ).simplify()
    assert real_ham.n_terms == 2

    # Unitary fourier transform.
    cr_def = (c_dag[a], dr.sum(
        (k, p.R), (1 / sqrt(n)) * exp(-I * 2 * pi * k * a / n) * c_dag[k]
    ))
    an_def = (c_[a], dr.sum(
        (k, p.R), (1 / sqrt(n)) * exp(I * 2 * pi * k * a / n) * c_[k]
    ))
    rec_ham = real_ham.subst_all([cr_def, an_def])
    res = rec_ham.simplify()

    assert res.n_terms == 1
    res_term = res.local_terms[0]
    assert len(res_term.sums) == 1
    dumm = res_term.sums[0][0]
    assert res_term.sums[0][1] == p.R
    # Here we mostly check the Hamiltonian has been diagonalized.
    assert len(res_term.vecs) == 2
    for i in res_term.vecs:
        assert len(i.indices) == 2
        assert i.indices[1] == dumm
