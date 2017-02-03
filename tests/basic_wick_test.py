"""Test basic Wick expansion of terms.

The tests in the module is attempted to test the core Wick facility on terms,
without parallelization by Spark.
"""

from sympy import symbols, IndexedBase

from drudge import Range, Vec, CR, AN, FERMI, FockDrudge
from drudge.term import sum_term
from drudge.wick import wick_expand


def test_wick_expansion_of_term(spark_ctx):
    """Test the basic Wick expansion facility on a single term."""

    dr = FockDrudge(spark_ctx, exch=FERMI)

    op_base = Vec('f')
    f = op_base[AN]
    f_dag = op_base[CR]

    a, b, c, d = symbols('a b c d')
    r = Range('L')
    t = IndexedBase('t')
    u = IndexedBase('u')

    term = sum_term(
        (a, r), (b, r), (c, r), (d, r),
        t[a, b] * u[c, d] * f_dag[a] * f[b] * f_dag[c] * f[d]
    )[0]

    res = wick_expand(
        term, comparator=dr.comparator, contractor=dr.contractor,
        phase=dr.phase
    )
    assert len(res) == 2

    # Simplify the result a little.

    dumms = {r: [a, b, c, d]}

    res = {
        i.simplify_deltas([lambda x: r])
            .canon(vec_colour=dr.vec_colour)
            .reset_dumms(dumms)[0]
        for i in res
        }

    expected = {
        sum_term(
            (a, r), (b, r), (c, r), (d, r),
            t[a, c] * u[b, d] * f_dag[a] * f_dag[b] * f[d] * f[c]
        )[0],
        sum_term(
            (a, r), (b, r), (c, r),
            t[a, c] * u[c, b] * f_dag[a] * f[b]
        )[0]
    }

    assert (res == expected)
