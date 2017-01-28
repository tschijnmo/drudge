"""Small utilities."""

from collections.abc import Sequence

from sympy import (sympify, Symbol, Expr, SympifyError, count_ops,
                   default_sort_key)


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

    return (count_ops(expr), default_sort_key(expr))


#
# Misc utilities
# --------------
#

def ensure_pair(obj, role):
    """Ensures that the given object is a pair."""
    if not (isinstance(obj, Sequence) and len(obj) == 2):
        raise TypeError('Invalid {}: '.format(role), obj, 'expecting pair')
    return obj
