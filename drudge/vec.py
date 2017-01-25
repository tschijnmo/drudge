"""Vectors and utilities."""

import collections.abc

from sympy import sympify


class Vec:
    """Vectors.

    Vectors are the basic non-commutative quantities.  Its objects consist of
    an base and some indices.  The base is allowed to be any Python object,
    although small hashable objects, like string, are advised.  The indices
    are always sympified into SymPy expressions.

    Its objects can be created directly by giving the base and indices,
    or existing vector objects can be subscripted to get new ones.  The
    semantics is similar to Haskell functions.

    Note that users cannot directly assign to the attributes of this class.

    This class can be used by itself, it can also be subclassed for special
    use cases.

    """

    __slots__ = ['_base', '_indices']

    def __init__(self, base, indices=()):
        """Initialize a vector.

        Atomic indices are added as the only index.  Iterable values will
        have all of its entries added.
        """
        self._base = base
        if not isinstance(indices, collections.abc.Iterable):
            indices = (indices,)
        self._indices = tuple(sympify(i) for i in indices)

    @property
    def base(self):
        """Get the base of the vector."""
        return self._base

    @property
    def indices(self):
        """Get the indices of the vector."""
        return self._indices

    def __getitem__(self, item):
        """Append the given indices to the vector.

        When multiple new indices are to be given, they have to be given as a
        tuple.
        """

        if not isinstance(item, tuple):
            item = (item,)

        new_indices = tuple(sympify(i) for i in item)

        # Pay attention to subclassing.
        return type(self)(self.base, self.indices + new_indices)

    def __repr__(self):
        """Form repr string form the vector."""
        return ''.join([
            type(self).__name__, '(', repr(self.base), ', (',
            ', '.join(repr(i) for i in self.indices),
            '))'
        ])

    def __str__(self):
        """Form a more readable string representation."""

        return ''.join([
            str(self.base), '[', ', '.join(str(i) for i in self.indices), ']'
        ])

    def __hash__(self):
        """Compute the hash value of a vector."""
        return hash((self.base, self.indices))

    def __eq__(self, other):
        """Compares the equality of two vectors."""
        return (
            (isinstance(self, type(other)) or isinstance(other, type(self))) and
            self.base == other.base and self.indices == other.indices
        )
