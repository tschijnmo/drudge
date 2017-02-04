"""Test basic Wick expansion of terms.

The tests in the module is attempted to test the core Wick facility on terms,
without parallelization by Spark.
"""

import pickle

from sympy import symbols, IndexedBase, latex, KroneckerDelta

from drudge import Range, Vec, CR, AN, FERMI, FockDrudge
from drudge.term import sum_term
from drudge.utils import sympy_key
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
        i.simplify_deltas([lambda x: r]).canon(
            vec_colour=dr.vec_colour
        ).reset_dumms(dumms)[0]
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


def test_ancr_character_has_basic_properties():
    """Test the annihilation and creation character singleton values.

    This test case also serves as the test for the general concrete symbol
    utility class and meta-class.
    """

    # Its objects will be serialized and deserialized.
    n_cr, n_an = [pickle.loads(pickle.dumps(i)) for i in [CR, AN]]

    # We test both the original value and the deserialized values.
    for cr, an in [(CR, AN), (n_cr, n_an)]:
        # Printing, all kinds of printing.
        assert str(cr) == 'CR'
        assert str(an) == 'AN'
        assert repr(cr) == 'CranChar.CR'
        assert repr(an) == 'CranChar.AN'
        assert latex(cr) == r'\dag'
        assert latex(an) == ''

        # Ordering, in its original form and as SymPy key.
        assert cr == cr
        assert an == an
        assert cr < an
        assert not cr > an
        assert sympy_key(cr) == sympy_key(cr)
        assert sympy_key(an) == sympy_key(an)
        assert sympy_key(cr) < sympy_key(an)
        assert not sympy_key(cr) > sympy_key(an)

        # Subtraction, purpose is the handling in deltas.
        assert cr - cr == 0
        assert an - an == 0
        assert cr - an != 0
        assert an - cr != 0
        assert KroneckerDelta(cr, cr) == 1
        assert KroneckerDelta(an, an) == 1
        assert KroneckerDelta(cr, an) == 0
        assert KroneckerDelta(an, cr) == 0
