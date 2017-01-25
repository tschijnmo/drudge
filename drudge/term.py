"""Tensor term definition and utility."""

from collections.abc import Iterable, Sequence

from sympy import sympify, SympifyError


class Range:
    """A symbolic range that can be summed over.

    This class is for symbolic ranges that is going to be summed over in
    tensors.  Each range should have a label, and optionally lower and upper
    bounds, which should be both given or absent.  The bounds will not be
    directly used for symbolic computation, but rather designed for printers
    and conversion to SymPy summation.  Note that ranges are assumed to be
    atomic and disjoint.  Even in the presence of lower and upper bounds,
    unequal ranges are assumed to be disjoint.

    .. warning::

         Unequal ranges are always assumed to be disjoint.

    """

    __slots__ = [
        '_label',
        '_lower',
        '_upper'
    ]

    def __init__(self, label, lower=None, upper=None):
        """Initialize the symbolic range."""
        self._label = label
        self._lower = sympify(lower) if lower is not None else lower

        if self._lower is None:
            if upper is not None:
                raise ValueError('lower range has not been given.')
            else:
                self._upper = None
        else:
            if upper is None:
                raise ValueError('upper range has not been given.')
            else:
                self._upper = sympify(upper)

    @property
    def label(self):
        """Get the label of the range."""
        return self._label

    @property
    def lower(self):
        """Get the lower bound of the range."""
        return self._lower

    @property
    def upper(self):
        """Get the upper bound of the range."""
        return self._upper

    @property
    def args(self):
        """Get the arguments for range creation.

        When the bounds are present, we have a triple, or we have a singleton
        tuple of only the label.
        """

        if self._lower is not None:
            return self._label, self._lower, self._upper
        else:
            return self._label,

    def __hash__(self):
        """Hash the symbolic range."""
        return hash(self.args)

    def __eq__(self, other):
        """Compare equality of two ranges."""
        return isinstance(other, type(self)) and (
            self.args == other.args
        )

    def __repr__(self):
        """Form the representative string."""
        return ''.join([
            'Range(', ', '.join(repr(i) for i in self.args), ')'
        ])

    def __str__(self):
        """Form readable string representation."""
        return str(self._label)


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
        if not isinstance(indices, Iterable):
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


class Term:
    """Terms in tensor expression.

    This is the core class for storing symbolic tensor expressions.  The
    actual symbolic tensor type is just a shallow wrapper over a list of
    terms.  It is basically comprised of three fields, a list of summations,
    a SymPy expression giving the amplitude, and a list of non-commutative
    vectors.
    """

    __slots__ = [
        '_sums',
        '_amp',
        '_vecs'
    ]

    def __init__(self, sums, amp, vecs):
        """Initialize the tensor term.

        This entry point should be the final place to check user inputs.
        """

        if not isinstance(sums, Iterable):
            raise TypeError('Invalid summations, iterable expected: ', sums)
        checked_sums = []
        for i in sums:
            if not (isinstance(i, Sequence) and len(i) == 2):
                raise TypeError('Invalid summation entry, pair expected: ', i)
            try:
                dummy = sympify(i[0])
            except SympifyError:
                raise TypeError('Invalid dummy, not sympifiable: ', i[0])
            if not isinstance(i[1], Range):
                raise TypeError('Invalid range to sum over: ', i)
            checked_sums.append((dummy, i[1]))
            continue
        self._sums = tuple(checked_sums)

        self._amp = sympify(amp)

        checked_vecs = []
        if not isinstance(vecs, Iterable):
            raise TypeError('Invalid vectors, should be iterable: ', vecs)
        for i in vecs:
            if not isinstance(i, Vec):
                raise ValueError('Invalid vector: ', i)
            checked_vecs.append(i)
            continue
        self._vecs = tuple(checked_vecs)

    @property
    def sums(self):
        """Get the summations of the term."""
        return self._sums

    @property
    def amp(self):
        """Get the amplitude expression."""
        return self._amp

    @property
    def vecs(self):
        """Gets the vectors in the term."""
        return self._vecs

    @property
    def args(self):
        """The triple of summations, amplitude, and vectors."""
        return self._sums, self._amp, self._vecs

    def __hash__(self):
        """Compute the hash of the term."""
        return hash(self.args)

    def __eq__(self, other):
        """Evaluate the equality with another term."""
        return isinstance(other, type(self)) and self.args == other.args

    def __repr__(self):
        """Form the representative string of a term."""
        return 'Term(sums=[{}], amp={}, vecs=[{}])'.format(
            ', '.join(repr(i) for i in self._sums),
            repr(self._amp),
            ', '.join(repr(i) for i in self._vecs)
        )

    def __str__(self):
        """Form the readable string representation of a term."""
        dumms = ', '.join(str(i[0]) for i in self._sums)
        factors = [str(self._amp)]
        factors.extend(str(i) for i in self._vecs)
        return 'sum_{{{}}} {}'.format(dumms, ' '.join(factors))
