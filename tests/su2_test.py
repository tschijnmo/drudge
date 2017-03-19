"""Test for the SU2 drudge."""

from sympy import Rational, I

from drudge import SU2LatticeDrudge


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
