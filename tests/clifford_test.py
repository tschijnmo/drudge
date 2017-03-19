"""Test for the Clifford algebra drudge."""

from drudge import CliffordDrudge, Vec, inner_by_delta


def test_clifford_drudge_by_quaternions(spark_ctx):
    """Test basic functionality of Clifford drudge by quaternions.
    """

    dr = CliffordDrudge(
        spark_ctx, inner=lambda v1, v2: -inner_by_delta(v1, v2)
    )
    e_ = Vec('e')

    i_ = dr.sum(e_[2] * e_[3]).simplify()
    j_ = dr.sum(e_[3] * e_[1]).simplify()
    k_ = dr.sum(e_[1] * e_[2]).simplify()

    for i in [i_, j_, k_]:
        assert (i * i).simplify() == -1

    assert (i_ * j_ * k_).simplify() == -1

    assert (i_ * j_).simplify() == k_
    assert (j_ * k_).simplify() == i_
    assert (k_ * i_).simplify() == j_
