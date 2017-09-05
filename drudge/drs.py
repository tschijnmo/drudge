"""Support for drudge scripts."""

import collections

from sympy import Symbol, Indexed, IndexedBase


#
# Special classes for SymPy objects
# ---------------------------------
#

class DrsSymbol(Symbol):
    """Symbols used in drudge scripts.

    The drudge symbol needs to behave as similar to the actual symbol as
    possible, because it is possible that they are used for keys in
    dictionaries.
    """

    __slots__ = [
        '_drudge',
        '_orig'
    ]

    def __new__(cls, drudge, name):
        """Create a symbol object."""
        symb = super().__new__(cls, name)
        return symb

    def __init__(self, drudge, name):
        """Initialize the symbol object."""
        self._drudge = drudge
        self._orig = Symbol(name)

    def __eq__(self, other):
        """Make equality comparison."""
        return self._orig == other

    def __hash__(self):
        """Compute the hash."""
        return hash(self._orig)

    def _hashable_content(self):
        """Hashable content for SymPy usages."""
        return self._orig._hashable_content()

    @classmethod
    def class_key(cls):
        return Symbol.class_key()

    def __getitem__(self, indices):
        """Index the given symbol.

        In drudge scripts, all symbols are by itself indexed bases.
        """
        base = IndexedBase(self._orig)
        if isinstance(indices, collections.Sequence):
            return DrsIndexed(self._drudge, base, *indices)
        else:
            return DrsIndexed(self._drudge, base, indices)

    def __iter__(self):
        """Disable iterability of the symbol.

        Or a default implementation from ``__getitem__`` will be used,
        which makes the symbols unable to be used as subscripts for indexed
        objects.
        """
        raise TypeError('Drudge script symbol cannot be iterated over.')


class DrsIndexed(Indexed):
    """Indexed objects for drudge scripts."""

    __slots__ = [
        '_drudge',
        '_orig'
    ]

    def __new__(cls, drudge, base, *args, **kwargs):
        """Create an indexed object for drudge scripts."""
        indexed = super().__new__(cls, base, *args, **kwargs)
        return indexed

    def __init__(self, drudge, base, *args, **kwargs):
        """Initialize the indexed object."""
        self._drudge = drudge
        self._orig = Indexed(base, *args, **kwargs)

    def __eq__(self, other):
        """Make equality comparison."""
        return self._orig == other

    def __hash__(self):
        """Compute the hash."""
        return hash(self._orig)

    def _hashable_content(self):
        """Hashable content for SymPy usages."""
        return self._orig._hashable_content()

    @classmethod
    def class_key(cls):
        return Indexed.class_key()
