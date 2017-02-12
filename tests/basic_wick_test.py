"""Test basic Wick expansion of simple things.
"""

import pickle

from sympy import latex, KroneckerDelta

from drudge import CR, AN
from drudge.utils import sympy_key


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
