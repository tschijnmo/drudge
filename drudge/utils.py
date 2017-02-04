"""Small utilities."""

from collections.abc import Sequence

from pyspark import SparkContext
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


#
# Misc utilities
# --------------
#

def ensure_pair(obj, role):
    """Ensures that the given object is a pair."""
    if not (isinstance(obj, Sequence) and len(obj) == 2):
        raise TypeError('Invalid {}: '.format(role), obj, 'expecting pair')
    return obj
