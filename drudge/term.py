"""Tensor term definition and utility."""

import abc
import collections
import functools
import itertools
import operator
import typing
import warnings
from collections.abc import Iterable, Mapping, Callable, Sequence

from sympy import (
    sympify, Symbol, KroneckerDelta, Eq, solveset, S, Integer, Add, Mul,
    Indexed, IndexedBase, Expr, Basic, Pow, Wild, conjugate, Sum, Piecewise,
    Intersection
)
from sympy.core.sympify import CantSympify

from .canon import canon_factors
from .utils import (
    ensure_symb, ensure_expr, sympy_key, is_higher, NonsympifiableFunc, prod_
)

#
# Utility constants
# -----------------
#

_UNITY = Integer(1)
_NEG_UNITY = Integer(-1)
_NAUGHT = Integer(0)


#
# Fundamental classes
# --------------------
#


class Range:
    """A symbolic range that can be summed over.

    This class is for symbolic ranges that is going to be summed over in
    tensors.  Each range should have a label, and optionally lower and upper
    bounds, which should be both given or absent.  The label can be any hashable
    and ordered Python type.  The bounds will not be directly used for symbolic
    computation, but rather designed for printers and conversion to SymPy
    summation.  Note that ranges are assumed to be atomic and disjoint.  Even in
    the presence of lower and upper bounds, unequal ranges are assumed to be
    disjoint.

    .. warning::

        Equality comparison and hashing of ranges are *completely* based on the
        label, without consideration of the bounds.  So ranges with the same
        label but different bounds will be considered equal.  This is to
        facilitate the usage of a same dummy set for ranges with slight
        variation in the bounds.

    .. warning::

         Unequal ranges are always assumed to be disjoint.

    """

    __slots__ = [
        '_label',
        '_lower',
        '_upper'
    ]

    def __init__(self, label, lower=None, upper=None):
        """Initialize the symbolic range.
        """
        self._label = label
        self._lower = (
            ensure_expr(lower, 'lower bound') if lower is not None else lower
        )

        if self._lower is None:
            if upper is not None:
                raise ValueError('lower range has not been given.')
            else:
                self._upper = None
        else:
            if upper is None:
                raise ValueError('upper range has not been given.')
            else:
                self._upper = ensure_expr(upper, 'upper bound')

    @property
    def label(self):
        """The label of the range."""
        return self._label

    @property
    def lower(self):
        """The lower bound of the range."""
        return self._lower

    @property
    def upper(self):
        """The upper bound of the range."""
        return self._upper

    @property
    def size(self):
        """The size of the range.

        This property given None for unbounded ranges.  For bounded ranges, it
        is the difference between the lower and upper bound.  Note that this
        contradicts the deeply entrenched mathematical convention of including
        other ends for a range.  But it does gives a lot of convenience and
        elegance.
        """

        return self._upper - self._lower if self.bounded else None

    @property
    def bounded(self):
        """If the range is explicitly bounded."""
        return self._lower is not None

    @property
    def args(self):
        """The arguments for range creation.

        When the bounds are present, we have a triple, or we have a singleton
        tuple of only the label.
        """

        if self.bounded:
            return self._label, self._lower, self._upper
        else:
            return self._label,

    def __hash__(self):
        """Hash the symbolic range.
        """
        return hash(self._label)

    def __eq__(self, other):
        """Compare equality of two ranges.
        """
        return isinstance(other, type(self)) and (
                self._label == other.label
        )

    def __repr__(self):
        """Form the representative string.
        """
        return ''.join([
            'Range(', ', '.join(repr(i) for i in self.args), ')'
        ])

    def __str__(self):
        """Form readable string representation.
        """
        return str(self._label)

    @property
    def sort_key(self):
        """The sort key for the range."""
        key = [self._label]
        if self.bounded:
            key.extend(sympy_key(i) for i in [self._lower, self._upper])
        return key

    def replace_label(self, new_label):
        """Replace the label of a given range.

        The bounds will be the same as the original range.
        """

        return Range(new_label, self._lower, self._upper)

    def __lt__(self, other):
        """Compare two ranges.

        This method is meant to skip explicit calling of the sort key when it is
        not convenient.
        """

        if not isinstance(other, Range):
            raise TypeError('Invalid range to compare', other)

        return self.sort_key < other.sort_key


class ATerms(abc.ABC):
    """Abstract base class for terms.

    This abstract class is meant for things that can be interpreted as a local
    collection of some tensor terms, mostly used for user input of tensor terms.

    """

    @property
    @abc.abstractmethod
    def terms(self) -> typing.List['Term']:
        """Get an list for the terms.
        """
        pass

    #
    # Mathematical operations.
    #

    _op_priority = 19.0  # Just less than the full tensor.

    def __mul__(self, other):
        """Multiply something on the right."""

        if is_higher(other, self._op_priority):
            return NotImplemented
        return self._mul(self.terms, parse_terms(other))

    def __rmul__(self, other):
        """Multiply something on the left."""

        if is_higher(other, self._op_priority):
            return NotImplemented
        return self._mul(parse_terms(other), self.terms)

    @staticmethod
    def _mul(left_terms, right_terms):
        """Multiply the left terms with the right terms.

        Note that the terms should not have any conflict in dummies.  Actually,
        by the common scheme in user input by drudge, the terms should normally
        have no summations at all.  So this function has different semantics
        than the term multiplication function from the Terms class.
        """

        prod_terms = []
        for i, j in itertools.product(left_terms, right_terms):
            # A shallow checking on sums, normally we have no sums by design.
            sums = _cat_sums(i.sums, j.sums)
            amp = i.amp * j.amp
            vecs = i.vecs + j.vecs

            prod_terms.append(Term(sums, amp, vecs))
            continue

        return Terms(prod_terms)

    def __add__(self, other):
        """Add something on the right."""

        if is_higher(other, self._op_priority):
            return NotImplemented
        return self._add(self.terms, parse_terms(other))

    def __radd__(self, other):
        """Add something on the left."""

        if is_higher(other, self._op_priority):
            return NotImplemented
        return self._add(parse_terms(other), self.terms)

    def __sub__(self, other):
        """Subtract something on the right."""

        if is_higher(other, self._op_priority):
            return NotImplemented
        other_terms = self._neg_terms(parse_terms(other))
        return self._add(self.terms, other_terms)

    def __rsub__(self, other):
        """Be subtracted from something on the left."""

        if is_higher(other, self._op_priority):
            return NotImplemented
        self_terms = self._neg_terms(parse_terms(self))
        return self._add(parse_terms(other), self_terms)

    def __neg__(self):
        """Negate the terms."""
        return Terms(self._neg_terms(parse_terms(self)))

    @staticmethod
    def _add(left_terms, right_terms):
        """Add the terms together.
        """
        return Terms(itertools.chain(left_terms, right_terms))

    @staticmethod
    def _neg_terms(terms: typing.Iterable['Term']):
        """Negate the given terms.

        The resulted terms are lazily evaluated.
        """
        return (
            Term(i.sums, i.amp * _NEG_UNITY, i.vecs)
            for i in terms
        )

    def __truediv__(self, other):
        """Make division with another object."""
        other = sympify(other)
        return Terms([
            i.scale(1 / other) for i in self.terms
        ])

    def __rtruediv__(self, other):
        """Being divided over by other object."""
        raise NotImplementedError('General tensors cannot inversed')

    def __or__(self, other):
        """Compute the commutator with another object.
        """
        if is_higher(other, self._op_priority):
            return NotImplemented
        return self * other - other * self

    def __ror__(self, other):
        """Compute the commutator with another object on the right."""
        if is_higher(other, self._op_priority):
            return NotImplemented
        return other * self - self * other


class Terms(ATerms):
    """A local collection of terms.

    This class is a concrete collection of terms.  Any mathematical operation on
    the abstract terms objects will be elevated to instances of this class.
    """

    __slots__ = ['_terms']

    def __init__(self, terms: typing.Iterable['Term']):
        """Initialize the terms object.

        The possibly lazy iterable of terms will be instantiated here.  And zero
        terms will be filtered out.
        """
        self._terms = list(i for i in terms if i.amp != 0)

    @property
    def terms(self):
        """Get the terms in the collection."""
        return self._terms


def parse_terms(obj) -> typing.List['Term']:
    """Parse the object into a list of terms."""

    if isinstance(obj, ATerms):
        return obj.terms
    else:
        expr = ensure_expr(obj)
        return [Term((), expr, ())]


class Vec(ATerms, CantSympify):
    """Vectors.

    Vectors are the basic non-commutative quantities.  Its objects consist of an
    label for its base and some indices.  The label is allowed to be any
    hashable and ordered Python object, although small objects, like string, are
    advised.  The indices are always sympified into SymPy expressions.

    Its objects can be created directly by giving the label and indices, or
    existing vector objects can be subscribed to get new ones.  The semantics is
    similar to Haskell functions.

    Note that users cannot directly assign to the attributes of this class.

    This class can be used by itself, it can also be subclassed for special
    use cases.

    Despite very different internal data structure, the this class is attempted
    to emulate the behaviour of the SymPy ``IndexedBase`` class

    """

    __slots__ = ['_label', '_indices']

    def __init__(self, label, indices=()):
        """Initialize a vector.

        Atomic indices are added as the only index.  Iterable values will
        have all of its entries added.
        """
        self._label = label
        if not isinstance(indices, Iterable):
            indices = (indices,)
        self._indices = tuple(ensure_expr(i, 'vector index') for i in indices)

    @property
    def label(self):
        """The label for the base of the vector.
        """
        return self._label

    @property
    def base(self):
        """The base of the vector.

        This base can be subscribed to get other vectors.
        """
        return Vec(self._label, [])

    @property
    def indices(self):
        """The indices to the vector.
        """
        return self._indices

    def __getitem__(self, item):
        """Append the given indices to the vector.

        When multiple new indices are to be given, they have to be given as a
        tuple.
        """

        if not isinstance(item, tuple):
            item = (item,)

        # Pay attention to subclassing.
        return type(self)(self._label, itertools.chain(self._indices, item))

    def __repr__(self):
        """Form repr string form the vector."""
        return ''.join([
            type(self).__name__, '(', repr(self._label), ', (',
            ', '.join(repr(i) for i in self._indices),
            '))'
        ])

    def __str__(self):
        """Form a more readable string representation."""

        label = str(self._label)
        if len(self._indices) > 0:
            indices = ''.join([
                '[', ', '.join(str(i) for i in self._indices), ']'
            ])
        else:
            indices = ''

        return label + indices

    def __hash__(self):
        """Compute the hash value of a vector."""
        return hash((self._label, self._indices))

    def __eq__(self, other):
        """Compares the equality of two vectors."""
        return (
                (isinstance(self, type(other)) or
                 isinstance(other, type(self))) and
                self._label == other.label and self._indices == other.indices
        )

    @property
    def sort_key(self):
        """The sort key for the vector.

        This is a generic sort key for vectors.  Note that this is only useful
        for sorting the simplified terms and should not be used in the
        normal-ordering operations.
        """

        key = [self._label]
        key.extend(sympy_key(i) for i in self._indices)
        return key

    #
    # Misc facilities
    #

    def map(self, func):
        """Map the given function to indices."""
        return Vec(self._label, (func(i) for i in self._indices))

    @property
    def terms(self):
        """Get the terms from the vector.

        This is for the user input.
        """

        return [Term((), _UNITY, (self,))]


class Term(ATerms):
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
        '_vecs',
        '_free_vars',
        '_dumms'
    ]

    def __init__(
            self, sums: typing.Tuple[typing.Tuple[Symbol, Range], ...],
            amp: Expr, vecs: typing.Tuple[Vec, ...],
            free_vars: typing.FrozenSet[Symbol] = None,
            dumms: typing.Mapping[Symbol, Range] = None,
    ):
        """Initialize the tensor term.

        Users seldom have the need to create terms directly by this function.
        So this constructor is mostly a developer function, no sanity checking
        is performed on the input for performance.  Most importantly, this
        constructor does **not** copy either the summations or the vectors and
        directly expect them to be tuples (for hashability).  And the amplitude
        is **not** simpyfied.

        Also, it is important that the free variables and dummies dictionary be
        given only when they really satisfy what we got for them.

        """

        # For performance reason, no checking is done.
        #
        # Uncomment for debugging.
        # valid = (
        #     isinstance(sums, tuple) and isinstance(amp, Expr)
        #     and isinstance(vecs, tuple)
        # )
        # if not valid:
        #     raise TypeError('Invalid argument to term creation')

        self._sums = sums
        self._amp = amp
        self._vecs = vecs

        self._free_vars = free_vars
        self._dumms = dumms

    @property
    def sums(self):
        """The summations of the term."""
        return self._sums

    @property
    def amp(self) -> Expr:
        """The amplitude expression."""
        return self._amp

    @property
    def vecs(self):
        """The vectors in the term."""
        return self._vecs

    @property
    def is_scalar(self):
        """If the term is a scalar."""
        return len(self._vecs) == 0

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

    @property
    def sort_key(self):
        """The sort key for a term.

        This key attempts to sort the terms by complexity, with simpler terms
        coming earlier.  This capability of sorting the terms will make the
        equality comparison of multiple terms easier.

        This sort key also ensures that terms that can be merged are always put
        into adjacent positions.

        """

        vec_keys = [i.sort_key for i in self._vecs]
        sum_keys = [(i[1].sort_key, sympy_key(i[0])) for i in self._sums]

        return (
            len(vec_keys), vec_keys,
            len(sum_keys), sum_keys,
            sympy_key(self._amp)
        )

    @property
    def terms(self):
        """The singleton list of the current term.

        This property is for the rare cases where direct construction of tensor
        inputs from SymPy expressions and vectors are not sufficient.
        """
        return [self]

    def scale(self, factor):
        """Scale the term by a factor.
        """
        return Term(self._sums, self._amp * factor, self._vecs)

    def mul_term(self, other, dumms=None, excl=None):
        """Multiply with another tensor term.

        Note that by this function, the free symbols in the two operands are not
        automatically excluded.
        """
        lhs, rhs = self.reconcile_dumms(other, dumms, excl)
        return Term(
            lhs.sums + rhs.sums, lhs.amp * rhs.amp, lhs.vecs + rhs.vecs
        )

    def comm_term(self, other, dumms=None, excl=None):
        """Commute with another tensor term.

        In ths same way as the multiplication operation, here the free symbols
        in the operands are not automatically excluded.
        """
        lhs, rhs = self.reconcile_dumms(other, dumms, excl)
        sums = lhs.sums + rhs.sums
        amp0 = lhs.amp * rhs.amp
        return [
            Term(sums, amp0, lhs.vecs + rhs.vecs),
            Term(sums, -amp0, rhs.vecs + lhs.vecs)
        ]

    def reconcile_dumms(self, other, dumms, excl):
        """Reconcile the dummies in two terms."""
        lhs, dummbegs = self.reset_dumms(dumms, excl=excl)
        rhs, _ = other.reset_dumms(dumms, dummbegs=dummbegs, excl=excl)
        return lhs, rhs

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
    def free_vars(self):
        """The free symbols used in the term.
        """

        if self._free_vars is None:
            dumms = self.dumms
            self._free_vars = set(
                i for expr in self.exprs for i in expr.atoms(Symbol)
                if i not in dumms
            )
        return self._free_vars

    @property
    def dumms(self):
        """Get the mapping from dummies to their range.
        """

        if self._dumms is None:
            self._dumms = dict(self._sums)
        return self._dumms

    @property
    def amp_factors(self):
        """The factors in the amplitude expression.

        This is a convenience wrapper over :py:meth:`get_amp_factors` for the
        case of no special additional symbols.
        """

        return self.get_amp_factors()

    def get_amp_factors(self, *special_symbs, monom_only=True):
        """Get the factors in the amplitude and the coefficient.

        The indexed factors and factors involving dummies or the symbols in the
        given special symbols set will be returned as a list, with the rest
        returned as a single SymPy expression.

        When ``monom_only`` is set, Error will be raised if the amplitude is not
        a monomial.
        """

        amp = self._amp

        if monom_only and isinstance(amp, Add):
            raise ValueError('Invalid amplitude: ', amp, 'expecting monomial')

        if isinstance(amp, Mul):
            all_factors = amp.args
        else:
            all_factors = (amp,)

        dumms = self.dumms

        factors = []
        coeff = _UNITY
        for factor in all_factors:
            need_treatment = any(
                (i in dumms or any(i in j for j in special_symbs))
                for i in factor.atoms(Symbol)
            ) or factor.has(Indexed)
            if need_treatment:
                factors.append(factor)
            else:
                coeff *= factor
            continue

        return factors, coeff

    def map(
            self, func=lambda x: x, sums=None, amp=None, vecs=None,
            skip_vecs=False
    ):
        """Map the given function to the SymPy expressions in the term.

        The given function will **not** be mapped to the dummies in the
        summations.  When operations on summations are needed, a **tuple**
        for the new summations can be given.

        By the default function of the identity function, this function can also
        be used to replace the summation list, the amplitude expression, or the
        vector part.
        """

        return Term(
            self._sums if sums is None else sums,
            func(self._amp if amp is None else amp),
            tuple(
                i.map(func) if not skip_vecs else i
                for i in (self._vecs if vecs is None else vecs)
            )
        )

    def subst(self, substs, sums=None, amp=None, vecs=None, purge_sums=False):
        """Perform symbol substitution on the SymPy expressions.

        After the replacement of the fields given, the given substitutions are
        going to be performed using SymPy ``xreplace`` method simultaneously.

        If purge sums is set, the summations whose dummy is substituted is going
        to be removed.

        """

        if sums is None:
            sums = self._sums
        if purge_sums:
            sums = tuple(i for i in sums if i[0] not in substs)

        return self.map(
            lambda x: x.xreplace(substs), sums=sums, amp=amp, vecs=vecs
        )

    def reset_dumms(self, dumms, dummbegs=None, excl=None, add_substs=None):
        """Reset the dummies in the term.

        The term with dummies reset will be returned alongside with the new
        dummy begins dictionary.  Note that the dummy begins dictionary will be
        mutated if one is given.

        ValueError will be raised when no more dummies are available.
        """

        new_sums, substs, dummbegs = self.reset_sums(
            self._sums, dumms, dummbegs, excl
        )
        if add_substs is not None:
            substs.update(add_substs)

        return self.subst(substs, new_sums), dummbegs

    @staticmethod
    def reset_sums(sums, dumms, dummbegs=None, excl=None):
        """Reset the given summations.

        The new summation list, substitution dictionary, and the new dummy begin
        dictionary will be returned.
        """

        if dummbegs is None:
            dummbegs = {}

        new_sums = []
        substs = {}
        for dumm_i, range_i in sums:

            # For linter.
            new_dumm = None
            new_beg = None

            beg = dummbegs[range_i] if range_i in dummbegs else 0
            for i in itertools.count(beg):

                try:
                    tentative = dumms[range_i][i]
                except KeyError:
                    raise ValueError('Dummies for range', range_i,
                                     'is not given')
                except IndexError:
                    raise ValueError('Dummies for range', range_i, 'is used up')

                if excl is None or tentative not in excl:
                    new_dumm = tentative
                    new_beg = i + 1
                    break
                else:
                    continue

            new_sums.append((new_dumm, range_i))
            substs[dumm_i] = new_dumm
            dummbegs[range_i] = new_beg

            continue

        return tuple(new_sums), substs, dummbegs

    #
    # Amplitude simplification
    #

    def simplify_deltas(self, resolvers):
        """Simplify deltas in the amplitude of the expression."""

        new_amp, substs = simplify_deltas_in_expr(
            self.dumms, self._amp, resolvers
        )

        # Note that here the substitutions needs to be performed in order.
        return self.subst(substs, purge_sums=True, amp=new_amp)

    def simplify_trivial_sums(self):
        """Simplify the trivial summations in the term.

        Trivial summations are the concrete summations that do not have any
        involvement at all.
        """

        involved = {
            i for expr in self.exprs for i in expr.atoms(Symbol)
        }

        new_sums = []
        factor = _UNITY
        dirty = False

        for symb, range_ in self._sums:
            if symb not in involved and range_.bounded:
                dirty = True
                factor *= range_.size
            else:
                new_sums.append((symb, range_))
            continue

        if dirty:
            return Term(tuple(new_sums), factor * self._amp, self._vecs)
        else:
            return self

    def expand(self):
        """Expand the term into many terms."""
        expanded_amp = self.amp.expand()

        if expanded_amp == 0:
            return []
        elif isinstance(expanded_amp, Add):
            amp_terms = expanded_amp.args
        else:
            amp_terms = (expanded_amp,)

        return [self.map(lambda x: x, amp=i) for i in amp_terms]

    #
    # Canonicalization.
    #

    def canon(self, symms=None, vec_colour=None):
        """Canonicalize the term.

        The given vector colour should be a callable accepting the index
        within vector list (under the keyword ``idx``) and the vector itself
        (under keyword ``vec``).  By default, vectors has colour the same as
        its index within the list of vectors.

        Note that whether or not colours for the vectors are given, the vectors
        are never permuted in the result.
        """

        if symms is None:
            symms = {}

        # Factors to canonicalize.
        factors = []

        # Additional information for factor reconstruction.
        #
        # It has integral placeholders for vectors and scalar factors without
        # any indexed quantity, the expression with (the only) indexed replaced
        # by the placeholder for factors with indexed.
        factors_info = []
        vec_factor = 1
        unindexed_factor = 2

        #
        # Get the factors in the amplitude.
        #

        # Cache globals for performance.
        wrapper_base = _WRAPPER_BASE
        treated_placeholder = _TREATED_PLACEHOLDER

        # Extractors for subexpressions requiring explicit treatment, defined
        # here to avoid repeated list and function creation for each factor.
        to_treats = []

        def replace_to_treats(expr: Expr):
            """Replace the quantities need treatment inside the factor.

            These quantities are explicitly indexed quantities and multi-valence
            functions either with explicit symmetry given or with multiple
            arguments involving dummies.
            """

            if_treat = False
            if expr.func == Indexed:
                if_treat = True
            elif len(expr.args) > 1:
                if expr.func in symms or (expr.func, len(expr.args)) in symms:
                    if_treat = True
                else:
                    if_treat = sum(1 for arg in expr.args if any(
                        arg.has(dumm) for dumm, _ in self.sums
                    )) > 1

            if if_treat:
                to_treats.append(expr)
                return treated_placeholder
            else:
                return expr

        amp_factors, coeff = self.amp_factors
        for i in amp_factors:

            amp_no_treated = i.replace(
                lambda _: True, NonsympifiableFunc(replace_to_treats)
            )

            n_to_treats = len(to_treats)
            if n_to_treats > 1:
                raise ValueError(
                    'Overly complex factor', i
                )
            elif n_to_treats == 1:
                to_treat = to_treats[0]
                to_treats.clear()  # Clean the container for the next factor.

                factors.append((
                    to_treat, (
                        _COMMUTATIVE,
                        sympy_key(to_treat.base.label)
                        if isinstance(to_treat, Indexed)
                        else sympy_key(to_treat.func),
                        sympy_key(amp_no_treated)
                    )
                ))
                factors_info.append(amp_no_treated)

            else:  # No part needing explicit treatment.

                # When the factor never has an indexed base, we treat it as
                # indexing a uni-valence internal indexed base.
                factors.append((
                    wrapper_base[i], (_COMMUTATIVE,)
                ))
                factors_info.append(unindexed_factor)

            continue

        #
        # Get the factors in the vectors.
        #

        for i, v in enumerate(self._vecs):
            colour = i if vec_colour is None else vec_colour(
                idx=i, vec=v, term=self
            )
            factors.append((
                v, (_NON_COMMUTATIVE, colour)
            ))
            factors_info.append(vec_factor)
            continue

        #
        # Invoke the core simplification.
        #

        res_sums, canoned_factors, canon_coeff = canon_factors(
            self._sums, factors, symms
        )

        #
        # Reconstruct the result
        #

        res_amp = coeff * canon_coeff
        res_vecs = []
        for i, j in zip(canoned_factors, factors_info):

            if j == vec_factor:
                # When we have a vector.
                res_vecs.append(i)
            elif j == unindexed_factor:
                res_amp *= i.indices[0]
            else:
                res_amp *= j.xreplace({treated_placeholder: i})
            continue

        return Term(tuple(res_sums), res_amp, tuple(res_vecs))

    def canon4normal(self, symms):
        """Canonicalize the term for normal-ordering.

        This is the preparation task for normal ordering.  The term will be
        canonicalized with all the vectors considered the same.  And the dummies
        will be reset internally according to the summation list.
        """

        # Make the internal dummies factory to canonicalize the vectors.
        dumms = collections.defaultdict(list)
        for i, v in self.sums:
            dumms[v].append(i)
        for i in dumms.values():
            # This is important for ordering vectors according to SymPy key in
            # normal ordering.
            i.sort(key=sympy_key)

        canon_term = (
            self.canon(symms=symms, vec_colour=lambda idx, vec, term: 0)
                .reset_dumms(dumms)[0]
        )

        return canon_term

    def has_base(self, base):
        """Test if the given base is present in the current term."""

        if isinstance(base, (IndexedBase, Symbol)):
            return self._amp.has(base)
        elif isinstance(base, Vec):
            label = base.label
            return any(
                i.label == label for i in self._vecs
            )
        else:
            raise TypeError('Invalid base to test presence', base)


_WRAPPER_BASE = IndexedBase(
    'internalWrapper', shape=('internalShape',)
)
_TREATED_PLACEHOLDER = Symbol('internalTreatedPlaceholder')

# For colour of factors in a term.

_COMMUTATIVE = 1
_NON_COMMUTATIVE = 0


#
# Substitution by tensor definition
# ---------------------------------
#

def subst_vec_term(
        term: Term, lhs: typing.Tuple[Vec], rhs_terms: typing.List[Term],
        dumms, dummbegs, excl
):
    """Substitute a matching vector in the given term.
    """

    sums = term.sums
    vecs = term.vecs
    amp = term.amp
    n_new_vecs = len(lhs)

    for i in range(len(vecs) - n_new_vecs + 1):
        substs = {}

        # Here we do simple hay-needle, KMP can be very hard here because of the
        # complexity of the predicate.
        for target, pattern in zip(vecs[i:], lhs):
            new = _match_indices(target, pattern)
            if new is None or not _update_substs(substs, new):
                break
            else:
                continue
        else:
            start_idx = i
            break

        continue
    else:
        return None  # Based on nest bind protocol.

    subst_states = _prepare_subst_states(
        rhs_terms, substs, dumms, dummbegs, excl
    )

    res = []
    for i, j in subst_states:
        new_vecs = list(vecs)
        new_vecs[start_idx:start_idx + n_new_vecs] = i.vecs
        res.append((
            Term(sums + i.sums, amp * i.amp, tuple(new_vecs)), j
        ))
        continue

    return res


def subst_factor_term(
        term: Term, lhs, rhs_terms: typing.List[Term], dumms, dummbegs, excl,
        full_simplify=True
):
    """Substitute a scalar factor in the term.

    While vectors are always flattened lists of vectors.  The amplitude part can
    be a lot more complex.  Here we strive to replace only one instance of the
    LHS by two placeholders, the substitution is possible only if the result
    expands to two terms, each containing only one of the placeholders.
    """

    amp = term.amp

    placeholder1 = Symbol('internalSubstPlaceholder1')
    placeholder2 = Symbol('internalSubstPlaceholder2')
    found = [False]
    substs = {}

    if isinstance(lhs, Symbol):
        label = lhs

        def query_func(expr):
            """Filter for the given symbol."""
            return not found[0] and expr == lhs

        def replace_func(_):
            """Replace the symbol."""
            found[0] = True
            return placeholder1 + placeholder2

    elif isinstance(lhs, Indexed):
        label = lhs.base.label

        # Here, in order to avoid the matching being called twice, we separate
        # the actual checking into both the query and the replace call-back.

        def query_func(expr):
            """Query for a reference to a given indexed base."""
            return not found[0] and isinstance(expr, Indexed)

        def replace_func(expr):
            """Replace the reference to the indexed base."""
            match_res = _match_indices(expr, lhs)
            if match_res is None:
                return expr
            found[0] = True
            assert len(substs) == 0
            substs.update(match_res)
            return placeholder1 + placeholder2

    else:
        raise TypeError(
            'Invalid LHS for substitution', lhs,
            'expecting symbol or indexed quantity'
        )

    # Some special treatment is needed for powers.
    pow_placeholder = Symbol('internalSubstPowPlaceholder')
    pow_val = [None]

    def decouple_pow(base, e):
        """Decouple a power."""
        if pow_val[0] is None and base.has(label):
            pow_val[0] = base
            return base * Pow(pow_placeholder, e - 1)
        else:
            return Pow(base, e)

    amp = amp.replace(Pow, NonsympifiableFunc(decouple_pow))
    amp = amp.replace(query_func, NonsympifiableFunc(replace_func))

    if not found[0]:
        return None

    if pow_val[0] is not None:
        amp = amp.xreplace({pow_placeholder: pow_val[0]})
    amp = (
        amp.simplify() if full_simplify else amp
    ).expand()

    # It is called nonlinear error, but some nonlinear forms, like conjugation,
    # can be handled.
    nonlinear_err = ValueError(
        'Invalid amplitude', term.amp, 'not expandable in', lhs
    )

    if not isinstance(amp, Add) or len(amp.args) != 2:
        raise nonlinear_err

    amp_term1, amp_term2 = amp.args
    diff = amp_term1.atoms(Symbol) ^ amp_term2.atoms(Symbol)
    if diff != {placeholder1, placeholder2}:
        raise nonlinear_err

    if amp_term1.has(placeholder1):
        amp = amp_term1
    else:
        amp = amp_term2

    subst_states = _prepare_subst_states(
        rhs_terms, substs, dumms, dummbegs, excl
    )

    sums = term.sums
    vecs = term.vecs
    res = []
    for i, j in subst_states:
        res.append((
            Term(sums + i.sums, amp.xreplace({placeholder1: i.amp}), vecs), j
        ))
        continue

    return res


def _match_indices(target, expr):
    """Match the target against the give expression for the indices.

    Both arguments must be scalar or vector indexed quantities.  The second
    argument should contain Wilds.
    """

    if target.base != expr.base or len(target.indices) != len(expr.indices):
        return None

    substs = {}
    for i, j in zip(target.indices, expr.indices):
        res = i.match(j)
        if res is None:
            return None
        else:
            if not _update_substs(substs, res):
                return None
            continue

    return substs


def _update_substs(substs, new):
    """Update the substitutions dictionary.

    If any of the new entry is in conflict with the old entry, a false will be
    returned, or we got true.
    """

    for k, v in new.items():
        if k not in substs:
            substs[k] = v
        elif v != substs[k]:
            return False
        continue

    return True


def _prepare_subst_states(rhs_terms, substs, dumms, dummbegs, excl):
    """Prepare the substitution states.

    Here we only have partially-finished substitution state for the next loop,
    where for each substituting term on the RHS, the given wild symbols in
    it will be substituted, then its dummies are going to be resolved.  Pairs of
    the prepared RHS terms and the corresponding dummbegs will be returned.  It
    is the responsibility of the caller to assemble the terms into the actual
    substitution state, by information in the term to be substituted.
    """

    subst_states = []
    for i, v in enumerate(rhs_terms):

        # Reuse existing dummy begins only for the first term.
        if i == 0:
            curr_dummbegs = dummbegs
        else:
            curr_dummbegs = dict(dummbegs)

        curr_term, curr_dummbegs = v.reset_dumms(dumms, curr_dummbegs, excl)
        subst_states.append((
            curr_term.subst(substs), curr_dummbegs
        ))
        continue

    return subst_states


def rewrite_term(
        term: Term, vecs: typing.Sequence[Vec], new_amp: Expr
) -> typing.Tuple[typing.Optional[Term], Term]:
    """Rewrite the given term.

    When a rewriting happens, the result will be the pair of the rewritten term
    and the term for the definition of the new amplitude, or the result will be
    None and the original term.
    """

    if len(term.vecs) != len(vecs):
        return None, term

    substs = {}
    for i, j in zip(term.vecs, vecs):
        curr_substs = _match_indices(i, j)
        if curr_substs is None:
            break
        for wild, expr in curr_substs.items():
            if wild in substs:
                if substs[wild] != expr:
                    break
            else:
                substs[wild] = expr
    else:
        # When a match is found.
        res_amp = new_amp.xreplace(substs)
        res_symbs = res_amp.atoms(Symbol)
        res_sums = tuple(i for i in term.sums if i[0] in res_symbs)
        def_sums = tuple(i for i in term.sums if i[0] not in res_symbs)
        return Term(res_sums, res_amp, term.vecs), Term(
            def_sums, term.amp, ()
        )

    return None, term


Sum_expander = typing.Callable[[Symbol], typing.Iterable[typing.Tuple[
    Symbol, Range, Expr
]]]


def expand_sums_term(term: Term, range_: Range, expander: Sum_expander):
    """Expand the given summations in the term.
    """

    res_sums = []
    dumms_to_expand = []
    for i in term.sums:
        if i[1] == range_:
            dumms_to_expand.append(i[0])
        else:
            res_sums.append(i)
        continue

    repl = {}
    for dumm in dumms_to_expand:
        for new_dumm, new_range, prev_form in expander(dumm):
            # Cleanse results from user callback.
            if not isinstance(new_dumm, Symbol):
                raise TypeError(
                    'Invalid dummy for the new summation', new_dumm
                )
            if not isinstance(new_range, Range):
                raise TypeError(
                    'Invalid range for the new summation', new_range
                )
            if not isinstance(prev_form, Expr):
                raise TypeError(
                    'Invalid previous form of the new summation dummy',
                    prev_form
                )
            res_sums.append((new_dumm, new_range))
            repl[prev_form] = new_dumm
            continue
        continue

    res_term = term.map(lambda x: x.xreplace(repl), sums=tuple(res_sums))

    def check_expr(expr: Expr):
        """Check if the given expression still has the expanded dummies."""
        for i in dumms_to_expand:
            if expr.has(i):
                raise ValueError(
                    'Scalar', expr, 'with original dummy', i
                )
            continue
        return expr

    res_term.map(check_expr)  # Discard result.
    return res_term


#
# User interface support
# ----------------------
#


def sum_term(sum_args, summand, predicate=None) -> typing.List[Term]:
    """Sum the given expression.

    This method is meant for easy creation of tensor terms.  The arguments
    should start with summations and ends with the expression that is summed.
    This core function is designed to be wrapped in functions working with full
    symbolic tensors.

    """

    # Too many SymPy stuff are callable.
    if isinstance(summand, Callable) and not isinstance(summand, Basic):
        inp_terms = None
        inp_func = summand
    else:
        inp_terms = parse_terms(summand)
        inp_func = None
        if len(sum_args) == 0:
            return list(inp_terms)

    sums, substs = _parse_sums(sum_args)

    res = []
    for sum_i in itertools.product(*sums):
        for subst_i in itertools.product(*substs):

            subst_dict = dict(subst_i)

            # We alway assemble the call sequence here, since this part should
            # never be performance critical.
            call_seq = dict(sum_i)
            call_seq.update(subst_dict)

            if not (predicate is None or predicate(call_seq)):
                continue

            if inp_terms is not None:
                curr_inp_terms = inp_terms
            else:
                curr_inp_terms = parse_terms(inp_func(call_seq))

            curr_terms = [i.subst(
                subst_dict, sums=_cat_sums(i.sums, sum_i)
            ) for i in curr_inp_terms]

            res.extend(curr_terms)

            continue
        continue

    return res


def _parse_sums(args):
    """Parse the summation arguments passed to the sum interface.

    The result will be the decomposed form of the summations and substitutions
    from the arguments.  For either of them, each entry in the result is a list
    of pairs of the dummy with the actual range or symbolic expression.
    """

    sums = []
    substs = []

    for arg in args:

        if not isinstance(arg, Sequence):
            raise TypeError('Invalid summation', arg, 'expecting a sequence')
        if len(arg) < 2:
            raise ValueError('Invalid summation', arg,
                             'expecting dummy and range')

        dumm = ensure_symb(arg[0], 'dummy')

        flattened = []
        for i in arg[1:]:
            if isinstance(i, Iterable):
                flattened.extend(i)
            else:
                flattened.append(i)
            continue

        contents = []
        expecting_range = None
        for i in flattened:
            if isinstance(i, Range):
                if expecting_range is None:
                    expecting_range = True
                elif not expecting_range:
                    raise ValueError('Invalid summation on', i,
                                     'expecting expression')
                contents.append((dumm, i))
            else:
                if expecting_range is None:
                    expecting_range = False
                elif expecting_range:
                    raise ValueError('Invalid summation on', i,
                                     'expecting a range')
                expr = ensure_expr(i)
                contents.append((dumm, expr))

        if expecting_range:
            sums.append(contents)
        else:
            substs.append(contents)

    return sums, substs


def _cat_sums(sums1, sums2):
    """Concatenate two summation lists.

    This function forms the tuple and ensures that there is no conflicting
    dummies in the two summations.  This function is mostly for sanitizing user
    inputs.
    """

    sums = tuple(itertools.chain(sums1, sums2))

    # Construction of the counter is separate from the addition of
    # content due to a PyCharm bug.
    dumm_counts = collections.Counter()
    dumm_counts.update(i[0] for i in sums)
    if any(i > 1 for i in dumm_counts.values()):
        raise ValueError(
            'Invalid summations to be concatenated', (sums1, sums2),
            'expecting no conflict in dummies'
        )

    return sums


def einst_term(term: Term, resolvers) -> typing.Tuple[
    typing.List[Term], typing.AbstractSet[Symbol]
]:
    """Add summations according to the Einstein convention to a term.

    The likely external indices for the term is also returned.
    """

    # Gather all indices to the indexed quantities.

    indices = []

    def proc_indexed(*args):
        """Get the indices to the indexed bases."""
        assert len(args) > 1
        indices.extend(args[1:])
        return Indexed(*args)

    term.amp.replace(Indexed, NonsympifiableFunc(proc_indexed))
    for i in term.vecs:
        indices.extend(i.indices)

    # Usage tally of the symbols, in bare form and in complex expressions.
    use_tally = collections.defaultdict(lambda: [0, 0])
    for index in indices:
        if isinstance(index, Symbol):
            use_tally[index][0] += 1
        else:
            for i in index.atoms(Symbol):
                use_tally[i][1] += 1
                continue
        continue

    existing_dumms = term.dumms
    new_sums = []
    exts = set()
    for symb, use in use_tally.items():

        if symb in existing_dumms:
            continue

        if use[1] != 0:
            # Non-conventional indices.
            continue

        if use[0] == 1:
            # External candidate.
            exts.add(symb)
        else:
            # Summation candidate.
            range_ = try_resolve_range(symb, {}, resolvers)
            if range_ is None:
                warnings.warn(
                    'Symbol {} is not summed for the incapability to resolve '
                    'range'
                        .format(symb)
                )
                continue

            # Now we have an Einstein summation.
            new_sums.append((symb, range_))
        continue

    # Make summation from Einstein convention deterministic.
    new_sums.sort(key=lambda x: (
        tuple(i.sort_key for i in (
            [x[1]] if isinstance(x[1], Range) else x[1]
        )),
        x[0].name
    ))

    return sum_term(new_sums, term), exts


def parse_term(term):
    """Parse a term.

    Other things that can be interpreted as a term are also accepted.
    """

    if isinstance(term, Term):
        return term
    elif isinstance(term, Vec):
        return Term((), _UNITY, (term,))
    else:
        return Term((), sympify(term), ())


#
# Delta simplification utilities.
# -------------------------------
#
# The core idea of delta simplification is that a delta can be replaced by a
# new, possibly simpler, expression, with a possible substitution on a dummy.
# The functions here aim to find and compose them.
#


def simplify_deltas_in_expr(sums_dict, amp, resolvers):
    """Simplify the deltas in the given expression.

    A new amplitude will be returned with all the deltas simplified, along with
    a dictionary giving the substitutions from the deltas.
    """

    substs = {}

    if amp == 0:
        return amp, substs

    # Preprocess some expressions equivalent to deltas into explicit delta form.
    arg0 = Wild('arg0')
    arg1 = Wild('arg1')
    value = Wild('value')
    amp = amp.replace(
        Piecewise((value, Eq(arg0, arg1)), (0, True)),
        KroneckerDelta(arg0, arg1) * value
    )

    new_amp = amp.replace(KroneckerDelta, NonsympifiableFunc(functools.partial(
        _proc_delta_in_amp, sums_dict, resolvers, substs
    )))

    return new_amp, substs


def compose_simplified_delta(amp, new_substs, substs, sums_dict, resolvers):
    """Compose delta simplification result with existing substitutions.

    This function can be interpreted as follows.  First we have a delta that has
    been resolved to be equivalent to an amplitude expression and some
    substitutions.  Then by this function, we get what it is equivalent to when
    we already have an existing bunch of earlier substitutions.

    The new substitutions should be given as an iterable of old/new pairs. Then
    the new amplitude and substitution from delta simplification can be composed
    with existing substitution dictionary.  New amplitude will be returned as
    the first return value. The given substitution dictionary will be mutated
    and returned as the second return value.  When the new substitution is
    incompatible with existing ones, the first return value will be a plain
    zero.

    The amplitude is a local thing in the expression tree, while the
    substitutions is always global among the entire term.  This function
    aggregate and expands it.

    """

    for subst in new_substs:
        if subst is None:
            continue
        old = subst[0]
        new = subst[1].xreplace(substs)

        to_add = None
        if old in substs:
            comp_amp, new_substs = proc_delta(
                substs[old], new, sums_dict, resolvers
            )
            amp = amp * comp_amp
            if new_substs is not None:
                # The new substitution cannot involve substituted symbols.
                to_add = {new_substs[0]: new_substs[1]}
                # amp could now be zero.
        else:
            # Easier case, a new symbol is tried to be added.
            to_add = {old: new}

        if to_add is not None:
            for i in substs.keys():
                substs[i] = substs[i].xreplace(to_add)
            substs.update(to_add)

        continue

    return amp, substs


def proc_delta(arg1, arg2, sums_dict, resolvers):
    """Processs a delta.

    An amplitude and a substitution pair is going to be returned.  The given
    delta will be equivalent to the returned amplitude factor with the
    substitution performed.  None will be returned for the substitution when no
    substitution is needed.
    """

    dumms = [
        i for i in set.union(arg1.atoms(Symbol), arg2.atoms(Symbol))
        if i in sums_dict
    ]

    if len(dumms) == 0:
        range1 = try_resolve_range(arg1, sums_dict, resolvers)
        range2 = try_resolve_range(arg2, sums_dict, resolvers)
        if range1 == range2:
            return KroneckerDelta(arg1, arg2), None
        else:
            return _NAUGHT, None

    eqn = Eq(arg1, arg2).simplify()

    # We try to solve for each of the dummies.  Most likely this will only be
    # executed for one loop.

    for dumm in dumms:
        range_ = sums_dict[dumm]
        # Here we assume the same integral domain, since dummies are summed over
        # and we can mostly assume that they are integral.
        #
        # TODO: infer actual domain from the range.
        domain = S.Integers
        sol = solveset(eqn, dumm, domain)

        # Strip off trivial intersecting with the domain.
        if isinstance(sol, Intersection) and len(sol.args) == 2:
            if sol.args[0] == domain:
                sol = sol.args[1]
            elif sol.args[1] == domain:
                sol = sol.args[0]

        if sol == domain:
            # Now we can be sure that we got an identity.
            return _UNITY, None
        elif len(sol) > 0:
            for i in sol:
                # Try to get the range of the substituting expression.
                range_of_i = try_resolve_range(i, sums_dict, resolvers)
                if range_of_i is None:
                    continue
                if range_of_i == range_:
                    return _UNITY, (dumm, i)
                else:
                    # We assume atomic and disjoint ranges!
                    return _NAUGHT, None
            # We cannot resolve the range of any of the solutions.  Try next
            # dummy.
            continue
        else:
            # No solution.
            return _NAUGHT, None

    # When we got here, all the solutions we found have undetermined range, we
    # have to return the unprocessed form.
    return KroneckerDelta(arg1, arg2), None


def _proc_delta_in_amp(sums_dict, resolvers, substs, *args):
    """Process a delta in the amplitude expression.

    The partial application of this function is going to be used as the
    call-back to SymPy replace function.  This function only returns SymPy
    expressions to satisfy SymPy replace interface.  All actions on the
    substitution are handled by an input/output argument.
    """

    # We first perform the substitutions found thus far.
    args = [i.xreplace(substs) for i in args]

    # Process the new delta.
    amp, subst = proc_delta(*args, sums_dict=sums_dict, resolvers=resolvers)

    new_amp, _ = compose_simplified_delta(
        amp, [subst], substs, sums_dict=sums_dict, resolvers=resolvers
    )

    return new_amp


#
# Amplitude summation simplification
# ----------------------------------
#


def simplify_amp_sums_term(term: Term, excl_bases, aggr, simplify):
    """Attempt to make simplifications to summations internal in amplitudes.
    """
    all_factors, coeff = term.get_amp_factors()
    if isinstance(coeff, Mul):
        all_factors.extend(coeff.args)
    else:
        all_factors.append(coeff)

    factors = []
    factor_symbs = []
    res_amp = 1  # Amplitude of the result to be incrementally built.
    imposs_symbs = set()  # Any symbol here will not be attempted.
    for factor in all_factors:
        symbs = factor.atoms(Symbol)
        if len(symbs) == 0:
            res_amp *= factor
            continue
        if excl_bases and isinstance(factor, Indexed):
            imposs_symbs |= symbs
            res_amp *= factor
            continue
        factors.append(factor)
        factor_symbs.append(symbs)
        continue
    n_factors = len(factors)

    for vec in term.vecs:
        for i in vec.indices:
            imposs_symbs |= i.atoms(Symbol)
            continue
        continue

    # From frozenset of indices of factors to the summation pairs.  Note that
    # the keys might have overlap, while the values should be disjoint.
    to_proc = collections.defaultdict(list)
    # Summations in the result to be incrementally built.
    res_sums = []

    for sum_ in term.sums:
        dumm, range_ = sum_
        if dumm in imposs_symbs or not range_.bounded:
            # Summations not intended to be treated here.
            res_sums.append(sum_)
            continue

        involving_factors = set()
        for i in range(n_factors):
            if dumm not in factor_symbs[i]:
                continue
            involving_factors.add(i)
            continue

        # Trivial summations should be treated already.
        assert len(involving_factors) > 0

        to_proc[frozenset(involving_factors)].append(sum_)
        continue

    proced_factors = set()  # Indices of factors already processed.

    # Main loop, try each bunch of summations with the same factor involvement
    # in turn.
    for curr_factors, curr_sums in to_proc.items():
        curr_expr = prod_(
            factors[i] for i in curr_factors
        )

        # Bitwise tricks for the power set.
        all_sums_mask = (1 << len(curr_sums)) - 1
        assert all_sums_mask > 0
        sums_mask = all_sums_mask

        while sums_mask == all_sums_mask if aggr else sums_mask > 0:
            sympy_sums = [
                (v[0], v[1].lower, v[1].upper - 1)
                for i, v in enumerate(curr_sums) if (1 << i & sums_mask) > 0
            ]
            orig = Sum(curr_expr, *sympy_sums)
            simplified = simplify(orig)

            if simplified is None or simplified == orig:
                sums_mask -= 1
                continue
            else:
                new_sums, new_factor = _apply_sum_simpl(
                    curr_sums, sums_mask, simplified
                )
                res_sums.extend(new_sums)
                res_amp *= new_factor
                proced_factors |= curr_factors
                break
        else:
            # No simplification has ever been found.
            res_sums.extend(curr_sums)

        continue

    # Place the unprocessed factors back.
    for i in range(n_factors):
        if i not in proced_factors:
            res_amp *= factors[i]
        continue

    return Term(sums=tuple(res_sums), amp=res_amp, vecs=term.vecs)


def _apply_sum_simpl(curr_sums, sums_mask, simplified):
    """Apply a particular summation simplification.

    The treated factors and summations will be equivalent to the returned
    symbolic summations and factor.
    """

    # The original symbolic summations.
    #
    # Attempted to be summed, but may wind up being untouched in the simplified
    # result.
    attempted = {}
    # The summations involved by the factors but are not included in the current
    # simplification.
    rest = []

    for i, v in enumerate(curr_sums):
        if 1 << i & sums_mask:
            assert v[0] not in attempted
            attempted[v[0]] = v[1]
        else:
            rest.append(v)
        continue

    if not isinstance(simplified, Sum):
        new_factor = simplified
    else:
        kept_sums = []  # Summations kept in SymPy amplitudes.
        for i in simplified.args[1:]:
            # Filter out summations which was, and can still be, handled by
            # drudge.
            if_untouched = (
                    len(i) == 3
                    and i[0] in attempted
                    and i[1] == attempted[i[0]].lower
                    and i[2] == attempted[i[0]].upper
            )
            if if_untouched:
                rest.append((i[0], attempted[i[0]]))
            else:
                kept_sums.append(i)
            continue

        if len(kept_sums) == 0:
            new_factor = simplified.args[0]
        else:
            new_factor = Sum(simplified.args[0], *kept_sums)

    return rest, new_factor


#
# Gradient computation
# --------------------
#

def diff_term(term: Term, variable, real, wirtinger_conj):
    """Differentiate a term.
    """

    symb = _GRAD_REAL_SYMB if real else _GRAD_SYMB

    if isinstance(variable, Symbol):

        lhs = variable
        rhs = lhs + symb

    elif isinstance(variable, Indexed):

        indices = variable.indices
        wilds = tuple(
            Wild(_GRAD_WILD_FMT.format(i)) for i, _ in enumerate(indices)
        )

        lhs = variable.base[wilds]
        rhs = lhs + functools.reduce(
            operator.mul,
            (KroneckerDelta(i, j) for i, j in zip(wilds, indices)), symb
        )

    else:
        raise ValueError('Invalid differentiation variable', variable)

    if real:
        orig_amp = term.amp.replace(conjugate(lhs), lhs)
    else:
        orig_amp = term.amp
    replaced_amp = (orig_amp.replace(lhs, rhs)).simplify()

    if real:
        eval_substs = {symb: 0}
    else:
        replaced_amp = replaced_amp.replace(
            conjugate(symb), _GRAD_CONJ_SYMB
        )
        eval_substs = {_GRAD_CONJ_SYMB: 0, symb: 0}

    if wirtinger_conj:
        diff_var = _GRAD_CONJ_SYMB
    else:
        diff_var = symb

    # Core evaluation.
    res_amp = replaced_amp.diff(diff_var).xreplace(eval_substs)
    res_amp = res_amp.simplify()

    return term.map(lambda x: x, amp=res_amp)


# Internal symbols for gradients.
_GRAD_SYMB_FMT = 'internalGradient{tag}Placeholder'
_GRAD_SYMB = Symbol(_GRAD_SYMB_FMT.format(tag=''))
_GRAD_CONJ_SYMB = Symbol(_GRAD_SYMB_FMT.format(tag='Conj'))
_GRAD_REAL_SYMB = Symbol(
    _GRAD_SYMB_FMT.format(tag='Real'), real=True
)

_GRAD_WILD_FMT = 'InternalWildSymbol{}'


#
# Misc public functions
# ---------------------
#

def try_resolve_range(i, sums_dict, resolvers):
    """Attempt to resolve the range of an expression.

    None will be returned if it cannot be resolved.
    """

    for resolver in itertools.chain([sums_dict], resolvers):

        if isinstance(resolver, Mapping):
            if i in resolver:
                return resolver[i]
            else:
                continue
        elif isinstance(resolver, Callable):
            range_ = resolver(i)
            if range_ is None:
                continue
            else:
                if isinstance(range_, Range):
                    return range_
                else:
                    raise TypeError('Invalid range: ', range_,
                                    'from resolver', resolver,
                                    'expecting range or None')
        else:
            raise TypeError('Invalid resolver: ', resolver,
                            'expecting callable or mapping')

    # Never resolved nor error found.
    return None
