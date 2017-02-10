"""Small utilities."""

import functools
import operator
from collections.abc import Sequence

from pyspark import RDD, SparkContext
from sympy import (
    sympify, Symbol, Expr, SympifyError, count_ops,
    default_sort_key, AtomicExpr, Integer, S
)
from sympy.core.assumptions import ManagedProperties


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


def is_higher(obj, priority):
    """Test if the object has higher operation priority.

    When the given object does not have defined priority, it is considered
    lower.
    """

    return getattr(obj, '_op_priority', priority - 1) > priority


class _EnumSymbsMeta(ManagedProperties):
    """The meta class for enumeration symbols.

    The primary purpose of this metaclass is to set the concrete singleton
    values from the enumerated symbols set in the class body.
    """

    SYMBS_INPUT = '_symbs_'

    def __new__(mcs, name, bases, attrs):
        """Create the new concrete symbols class."""

        cls = super().__new__(mcs, name, bases, attrs)

        if not hasattr(cls, mcs.SYMBS_INPUT):
            raise AttributeError('Cannot find attribute ' + mcs.SYMBS_INPUT)

        symbs = getattr(cls, mcs.SYMBS_INPUT)
        if symbs is None:
            # Base class.
            return cls

        if not isinstance(symbs, Sequence):
            raise ValueError('Invalid symbols', symbs, 'expecting a sequence')
        for i in symbs:
            invalid = not isinstance(i, Sequence) or len(i) != 2 or any(
                not isinstance(j, str) for j in i
            )
            if invalid:
                raise ValueError(
                    'Invalid symbol', i,
                    'expecting pairs of identifier and LaTeX form.'
                )
        if len(symbs) < 2:
            raise ValueError(
                'Invalid symbols ', symbs, 'expecting multiple of them'
            )

        for i, v in enumerate(symbs):
            obj = cls(i)
            setattr(cls, v[0], obj)
            continue

        return cls


class EnumSymbs(AtomicExpr, metaclass=_EnumSymbsMeta):
    """Base class for enumeration symbols.

    Subclasses can set `_symbs_` inside the class body to be a sequence of
    string pairs.  Then attributes named after the first field of the pairs will
    be created, with the LaTeX form controlled by the second pair.

    The resulted values are valid SymPy expressions.  They are ordered according
    to their order in the given enumeration sequence.

    """

    _symbs_ = None

    _VAL_FIELD = '_val_index'
    __slots__ = [_VAL_FIELD]

    def __init__(self, val_index):
        """Initialize the concrete symbol object.
        """
        if self._symbs_ is None:
            raise ValueError('Base EnumSymbs class cannot be instantiated')
        setattr(self, self._VAL_FIELD, val_index)

    @property
    def args(self):
        """The argument for SymPy."""
        return Integer(getattr(self, self._VAL_FIELD)),

    def __str__(self):
        """Get the string representation of the symbol."""
        return self._symbs_[getattr(self, self._VAL_FIELD)][0]

    def __repr__(self):
        """Get the machine readable string representation."""
        return '.'.join([type(self).__name__, str(self)])

    _op_priority = 20.0

    def __eq__(self, other):
        """Test two values for equality."""
        return isinstance(other, type(self)) and self.args == other.args

    def __hash__(self):
        """Hash the concrete symbol object."""
        return hash(repr(self))

    def __lt__(self, other):
        """Test two values for less than order.

        The order will be based on the order given in the class.
        """
        return self.args < other.args

    def __gt__(self, other):
        """Test two values for greater than."""
        return self.args > other.args

    def __sub__(self, other):
        """Subtract the current value with another.

        This method is mainly to be able to work together with the Kronecker
        delta class from SymPy.
        """

        if not isinstance(other, type(self)):
            raise ValueError(
                'Invalid operation for ', (self, other),
                'concrete symbols can only be subtracted for the same type'
            )

        return self.args[0] - other.args[0]

    def sort_key(self, order=None):
        return (
            self.class_key(),
            (1, tuple(i.sort_key() for i in self.args)),
            S.One.sort_key(), S.One
        )

    def _latex(self, _):
        """Print itself as LaTeX code."""
        return self._symbs_[self.args[0]][1]


#
# Spark utilities
# ---------------
#


class BCastVar:
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


def nest_bind(rdd: RDD, func):
    """Nest the flat map of the given function.

    When an entry no longer need processing, None can be returned by the call
    back function.

    """

    ctx = rdd.context

    def wrapped(obj):
        """Wrapped function for nest bind."""
        vals = func(obj)
        if vals is None:
            return [(False, obj)]
        else:
            return [(True, i) for i in vals]

    curr = rdd
    curr.cache()
    res = []
    while curr.count() > 0:
        step_res = curr.flatMap(wrapped)
        step_res.cache()
        new_entries = step_res.filter(lambda x: not x[0]).map(lambda x: x[1])
        new_entries.cache()
        res.append(new_entries)
        curr = step_res.filter(lambda x: x[0]).map(lambda x: x[1])
        curr.cache()
        continue

    return ctx.union(res)


def nest_bind_serial(data, func):
    """Nest the flat map of the given function serially.

    This function has the same semantics as the nest bind function for Spark
    RDD.  It is mainly for the purpose of testing and debugging.

    """

    curr = data
    res = []
    while len(curr) > 0:
        new_curr = []
        for i in curr:
            step_res = func(i)
            if step_res is None:
                res.append(i)
            else:
                new_curr.extend(step_res)
            continue
        curr = new_curr
        continue

    return res


#
# Misc utilities
# --------------
#

def ensure_pair(obj, role):
    """Ensures that the given object is a pair."""
    if not (isinstance(obj, Sequence) and len(obj) == 2):
        raise TypeError('Invalid {}: '.format(role), obj, 'expecting pair')
    return obj


#
# Small user utilities
# --------------------
#

def sum_(obj):
    """Sum the values in the given iterable.

    Different from the built-in summation function, here a value zero is created
    only when the iterator is empty.  Or the summation is based on the first
    item in the iterable.
    """

    i = iter(obj)
    try:
        init = next(i)
    except StopIteration:
        return 0
    else:
        return functools.reduce(operator.add, i, init)


def prod_(obj):
    """Product the values in the given iterable.

    Similar to the summation utility function, here the initial value for the
    reduction is the first element.  Different from the summation, here
    a integer unity will be returned for empty iterator.
    """

    i = iter(obj)
    try:
        init = next(i)
    except StopIteration:
        return 1
    else:
        return functools.reduce(operator.mul, i, init)
