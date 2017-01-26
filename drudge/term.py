"""Tensor term definition and utility."""

import itertools
import typing
from collections.abc import Iterable

from sympy import sympify, Symbol

from .utils import ensure_pair, ensure_symb, ensure_expr


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
        self._indices = tuple(ensure_expr(i, 'vector index') for i in indices)

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

    #
    # Multiplication
    #

    _op_priority = 20.0

    def __mul__(self, other):
        """Multiply something on the right."""

        if isinstance(other, Term):
            # Delegate to the term for the multiplication.
            return NotImplemented
        if isinstance(other, Vec):
            return Term([], 1, [self, other])
        else:
            return Term([], sympify(other), [self])

    def __rmul__(self, other):
        """Multiply something on the left."""

        # In principle, other should not be either a term or a vector.
        return Term([], sympify(other), [self])

    #
    # Misc facilities
    #

    def map(self, func):
        """Map the given function to indices."""
        return Vec(self._base, (func(i) for i in self._indices))


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
        dumms = set()
        for i in sums:
            i = ensure_pair(i, 'summation')
            dumm = ensure_symb(i[0], 'dummy')
            if dumm in dumms:
                raise ValueError('Invalid dummy: ', dumm, 'duplicated')
            if not isinstance(i[1], Range):
                raise TypeError('Invalid range: ', i[1], 'not Range instance')
            checked_sums.append((dumm, i[1]))
            continue
        self._sums = tuple(checked_sums)

        self._amp = sympify(amp)

        checked_vecs = []
        if not isinstance(vecs, Iterable):
            raise TypeError('Invalid vectors: ', vecs, 'expecting iterable')
        for i in vecs:
            if not isinstance(i, Vec):
                raise ValueError('Invalid vector: ', i, 'expecting Vec')
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
        if len(self._sums) > 0:
            header = 'sum_{{{}}} '.format(
                ', '.join(str(i[0]) for i in self._sums))
        else:
            header = ''
        factors = [str(self._amp)]
        factors.extend(str(i) for i in self._vecs)
        return header + ' * '.join(factors)

    #
    # Multiplication
    #

    _op_priority = 20.0

    def __mul__(self, other):
        """Multiple something on the right."""

        if isinstance(other, Term):
            # Now for tensor term creation, we do not need this yet
            #
            # TODO: Add implementaion.
            raise NotImplementedError()
        elif isinstance(other, Vec):
            return Term(self._sums, self._amp, self._vecs + (other,))
        else:
            return Term(self._sums, self._amp * sympify(other), self._vecs)

    def __rmul__(self, other):
        """Multiply something on the left."""

        # In principle, the other operand should not be another term.
        if isinstance(other, Vec):
            return Term(self._sums, self._amp, (other,) + self._vecs)
        else:
            return Term(self._sums, sympify(other) * self._amp, self._vecs)

    #
    # SymPy related
    #

    @property
    def exprs(self):
        """Loop over the sympy expression in the term.

        Note that the summation dummies are not looped over.
        """

        yield self._amp
        for vec in self._vecs:
            yield from vec.indices

    @property
    def symbs(self):
        """Get the symbols used in the term.

        The free and dummy symbols used in the term are going to be returned as
        two sets.
        """

        dumms = set(i[0] for i in self._sums)
        frees = set(i for expr in self.exprs for i in expr.atoms(Symbol)
                    if i not in dumms)
        return frees, dumms

    def map(self, func, sums=None):
        """Map the given function to the SymPy expressions in the term.

        The given function will **not** be mapped to the dummies in the
        summations.  When operations on summations are needed, an iterable
        for the new summations can be given.
        """

        return Term(
            self._sums if sums is None else sums,
            func(self._amp),
            (i.map(func) for i in self._vecs)
        )


def sum_term(*args, predicate=None) -> typing.List[Term]:
    """Sum the given expression.

    This method is meant for easy creation of tensor terms.  The arguments
    should start with summations and ends with the expression that is summed.

    The summations should be given as pairs, all with the first field being a
    SymPy symbol for summation.  The second field can be a symbolic range,
    for which the dummy is summed over.  Or an iterable can also be given,
    whose entries can be both symbolic ranges or SymPy expressions.

    The predicate can be a callable going to return a boolean when given a
    dictionary giving the action on each of the dummies.  False values
    can be used the skip some terms.

    This core function is designed to be wrapped in functions working with
    full symbolic tensors.

    """

    if len(args) == 0:
        return []
    elif len(args) == 1:
        return [args[0]]

    sums, substs = _parse_sums(args[:-1])

    inp_term = _parse_term(args[-1])

    res = []
    for sum_i in itertools.product(*sums):
        for subst_i in itertools.product(*substs):

            if predicate is not None:
                full_dict = dict(sum_i)
                full_dict.update(subst_i)
                if not predicate(full_dict):
                    continue

            res.append(inp_term.map(
                lambda x: x.subs(subst_i, simultaneous=True),
                itertools.chain(inp_term.sums, sum_i)
            ))

            continue
        continue

    return res


def _parse_sums(args):
    """Parse the summation arguments passed to the sum interface.

    The result will be the decomposed form of the summations and
    substitutions from the arguments.
    """

    sums = []
    substs = []

    for i in args:

        i = ensure_pair(i, 'summation')
        dumm = ensure_symb(i[0], 'dummy')

        if isinstance(i[1], Range):
            sums.append([(dumm, i[1])])
        else:
            if not isinstance(i[1], Iterable):
                raise TypeError(
                    'Invalid range: ', i[1], 'expecting range or iterable')
            entries = list(i[1])
            if len(entries) < 1:
                raise ValueError('Invalid summation range for ', dumm,
                                 'expecting non-empty iterable')
            if any(isinstance(j, Range) for j in entries):
                if all(isinstance(j, Range) for j in entries):
                    sums.append([(dumm, j) for j in entries])
                else:
                    raise TypeError('Invalid summation range: ', entries,
                                    'expecting all ranges')
            else:
                substs.append([(dumm, ensure_expr(j)) for j in entries])

    return sums, substs


def _parse_term(term):
    """Parse a term.

    Other things that can be interpreted as a term are also accepted.
    """

    if isinstance(term, Term):
        return term
    elif isinstance(term, Vec):
        return Term([], 1, [term])
    else:
        return Term([], term, [])
