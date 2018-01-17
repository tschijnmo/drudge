"""Test for the SU2 drudge."""

from sympy import Rational, I, Symbol, symbols, IndexedBase

from drudge import SU2LatticeDrudge, Range


def test_su2_without_symbolic_index(spark_ctx):
    """Test SU2 lattice drudge without abstract symbolic lattice index."""

    dr = SU2LatticeDrudge(spark_ctx)
    p = dr.names
    half = Rational(1, 2)
    half_i = half / I

    # Test the basic commutation rules without explicit site or on the same
    # site.
    for ops in [
        (p.J_, p.J_p, p.J_m),
        (p.J_[0], p.J_p[0], p.J_m[0])
    ]:
        j_z, j_p, j_m = [dr.sum(i) for i in ops]
        assert (j_z | j_p).simplify() == j_p
        assert (j_z | j_m).simplify() == -1 * j_m
        assert (j_p | j_m).simplify() == 2 * j_z

        j_x = (j_p + j_m) * half
        j_y = (j_p - j_m) * half_i
        assert (j_x | j_y).simplify() == I * j_z
        assert (j_y | j_z).simplify() == I * j_x
        assert (j_z | j_x).simplify() == I * j_y

        j_sq = dr.sum(
            j_z * j_z + half * j_p * j_m + half * j_m * j_p
        )
        for i in [j_x, j_y, j_z]:
            assert (j_sq | i).simplify() == 0
        continue


def test_su2_on_1d_heisenberg_model(spark_ctx):
    """Test the SU2 drudge on 1D Heisenberg model with abstract lattice indices.

    This test also acts as the test for the default resolver.
    """

    dr = SU2LatticeDrudge(spark_ctx)
    l = Range('L')
    dr.set_dumms(l, symbols('i j k l m n'))
    dr.add_default_resolver(l)

    p = dr.names
    j_z = p.J_
    j_p = p.J_p
    j_m = p.J_m
    i = p.i
    half = Rational(1, 2)

    coupling = Symbol('J')
    ham = dr.sum(
        (i, l),
        j_z[i] * j_z[i + 1] +
        j_p[i] * j_m[i + 1] / 2 + j_m[i] * j_p[i + 1] / 2
    ) * coupling

    s_sq = dr.sum(
        (i, l),
        j_z[i] * j_z[i] + half * j_p[i] * j_m[i] + half * j_m[i] * j_p[i]
    )

    comm = (ham | s_sq).simplify()
    assert comm == 0


def test_su2_with_deformed_commutation(spark_ctx):
    """Test SU2 lattice drudge with site-dependent commutation rules."""

    raise_ = SU2LatticeDrudge.DEFAULT_RAISE
    lower = SU2LatticeDrudge.DEFAULT_LOWER
    cartan = SU2LatticeDrudge.DEFAULT_CARTAN
    alpha = IndexedBase('alpha')
    a = Symbol('a')

    dr = SU2LatticeDrudge(spark_ctx, specials={
        (raise_[a], lower[a]): alpha[a] * cartan[a] - 1
    })

    assert dr.simplify(cartan[a] | raise_[a]) == dr.sum(raise_[a])
    assert dr.simplify(cartan[a] | lower[a]) == dr.sum(-lower[a])

    assert dr.simplify(raise_[a] | lower[a]) == dr.sum(
        alpha[a] * cartan[a] - 1
    ).simplify()
    assert dr.simplify(lower[a] | raise_[a]) == dr.sum(
        1 - alpha[a] * cartan[a]
    ).simplify()
    assert dr.simplify(raise_[1] | lower[2]) == 0
