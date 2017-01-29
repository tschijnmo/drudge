"""Small utilities."""

from collections.abc import Sequence

from sympy import (sympify, Symbol, Expr, SympifyError, count_ops,
                   default_sort_key)

from pyspark import SparkContext


#
# SymPy utilities
# ---------------
#

def ensure_sympify(obj, role='', expected_type=None):
    """Sympify the given object with checking and error reporting.

    This is a shallow wrapper over SymPy sympify function to have error
    reporting in consistent style and an optional type checking.
    """

    header = 'Invalid {}: '.format(role)

    try:
        sympified = sympify(obj)
    except SympifyError as exc:
        raise TypeError(header, obj, 'failed to be simpified', exc.args)

    if expected_type is None or isinstance(sympified, expected_type):
        return sympified
    else:
        raise TypeError(header, sympified, 'expecting', expected_type)


def ensure_symb(obj, role=''):
    """Sympify the given object into a symbol."""
    return ensure_sympify(obj, role, Symbol)


def ensure_expr(obj, role=''):
    """Sympify the given object into an expression."""
    return ensure_sympify(obj, role, Expr)


def sympy_key(expr):
    """Get the key for ordering SymPy expressions.

    This function assumes that the given expression is already sympified.
    """

    return count_ops(expr), default_sort_key(expr)


#
# Spark utilities
# ---------------
#


class BCast:
    """Automatically broadcast variables.

    This class is a shallow encapsulation of a variable and its broadcast
    into the spark context.  The variable can be redistributed automatically
    after any change.

    """

    __slots__ = [
        '_ctx',
        '_var',
        '_bcast'
    ]

    def __init__(self, ctx: SparkContext, var):
        """Initialize the broadcast variable."""
        self._ctx = ctx
        self._var = var
        self._bcast = None

    @property
    def var(self):
        """Get the variable to mutate."""
        self._bcast = None
        return self._var

    @property
    def ro(self):
        """Get the variable, read-only.

        Note that this function only prevents the redistribution of the
        variable.  It cannot force the variable not be mutated.
        """
        return self._var

    @property
    def bcast(self):
        """Get the broadcast variable."""
        if self._bcast is None:
            self._bcast = self._ctx.broadcast(self._var)
        return self._bcast


#
# Misc utilities
# --------------
#

def ensure_pair(obj, role):
    """Ensures that the given object is a pair."""
    if not (isinstance(obj, Sequence) and len(obj) == 2):
        raise TypeError('Invalid {}: '.format(role), obj, 'expecting pair')
    return obj
