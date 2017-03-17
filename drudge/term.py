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
    sympify, Symbol, KroneckerDelta, Eq, solve, S, Integer, Add, Mul, Indexed,
    IndexedBase, Expr, Basic, Pow, Wild, conjugate
)
from sympy.core.sympify import CantSympify

from .canon import canon_factors
from .utils import (
    ensure_symb, ensure_expr, sympy_key, is_higher, NonsympifiableFunc
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

        Bounds with the same label but different bounds will be considered
        unequal.  Although no error be given, using different bounds with
        identical label is strongly advised against.

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
        return hash(self.args)

    def __eq__(self, other):
        """Compare equality of two ranges.
        """
        return isinstance(other, type(self)) and (
            self.args == other.args
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

    @abc.abstractproperty
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
            (isinstance(self, type(other)) or isinstance(other, type(self))) and
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

        return self.get_amp_factors(set())

    def get_amp_factors(self, special_symbs):
        """Get the factors in the amplitude and the coefficient.

        The indexed factors and factors involving dummies or the symbols in the
        given special symbols set will be returned as a list, with the rest
        returned as a single SymPy expression.

        Error will be raised if the amplitude is not a monomial.
        """

        amp = self._amp

        if isinstance(amp, Add):
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
                (i in dumms or i in special_symbs)
                for i in factor.atoms(Symbol)
            ) or isinstance(factor, Indexed)
            if need_treatment:
                factors.append(factor)
            else:
                coeff *= factor
            continue

        return factors, coeff

    def map(self, func, sums=None, amp=None, vecs=None, skip_vecs=False):
        """Map the given function to the SymPy expressions in the term.

        The given function will **not** be mapped to the dummies in the
        summations.  When operations on summations are needed, a **tuple**
        for the new summations can be given.

        By passing the identity function, this function can also be used to
        replace the summation list, the amplitude expression, or the vector
        part.
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

    def reset_dumms(self, dumms, dummbegs=None, excl=None):
        """Reset the dummies in the term.

        The term with dummies reset will be returned alongside with the new
        dummy begins dictionary.  Note that the dummy begins dictionary will be
        mutated if one is given.

        ValueError will be raised when no more dummies are available.
        """

        new_sums, substs, dummbegs = self.reset_sums(
            self._sums, dumms, dummbegs, excl
        )

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
        indexed_placeholder = _INDEXED_PLACEHOLDER

        # Extractors for the indexed, defined here to avoid repeated list and
        # function creation for each factor.
        indexed = []

        def replace_indexed(base, *indices):
            """Replace the indexed quantity inside the factor."""
            indexed.append(base[indices])
            return indexed_placeholder

        amp_factors, coeff = self.amp_factors
        for i in amp_factors:

            amp_no_indexed = i.replace(
                Indexed, NonsympifiableFunc(replace_indexed)
            )

            n_indexed = len(indexed)
            if n_indexed > 1:
                raise ValueError(
                    'Invalid amplitude factor containing multiple indexed', i
                )
            elif n_indexed == 1:

                factors.append((
                    indexed[0], (
                        _COMMUTATIVE,
                        indexed[0].base.label.name,
                        sympy_key(amp_no_indexed)
                    )
                ))
                factors_info.append(amp_no_indexed)

                indexed.clear()  # Clean the container for the next factor.

            else:  # No indexed.

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
            self._sums, factors, symms if symms is not None else {}
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
                res_amp *= j.xreplace({indexed_placeholder: i})
            continue

        return Term(tuple(res_sums), res_amp, tuple(res_vecs))

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
_INDEXED_PLACEHOLDER = Symbol('internalIndexedPlaceholder')

# For colour of factors in a term.

_COMMUTATIVE = 1
_NON_COMMUTATIVE = 0


#
# Substitution by tensor definition
# ---------------------------------
#

def subst_vec_in_term(term: Term, lhs: Vec, rhs_terms: typing.List[Term],
                      dumms, dummbegs, excl):
    """Substitute a matching vector in the given term.
    """

    sums = term.sums
    vecs = term.vecs
    amp = term.amp

    for i, v in enumerate(vecs):
        substs = _match_indices(v, lhs)
        if substs is None:
            continue
        else:
            substed_vec_idx = i
            break
    else:
        return None  # Based on nest bind protocol.

    subst_states = _prepare_subst_states(
        rhs_terms, substs, dumms, dummbegs, excl
    )

    res = []
    for i, j in subst_states:
        new_vecs = list(vecs)
        new_vecs[substed_vec_idx:substed_vec_idx + 1] = i.vecs
        res.append((
            Term(sums + i.sums, amp * i.amp, tuple(new_vecs)), j
        ))
        continue

    return res


def subst_factor_in_term(term: Term, lhs, rhs_terms: typing.List[Term],
                         dumms, dummbegs, excl, full_simplify=True):
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
        amp = amp.subs(pow_placeholder, pow_val[0])
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
            Term(sums + i.sums, amp.subs(placeholder1, i.amp), vecs), j
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
            substs.update(res)
            continue

    return substs


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


def einst_term(term: Term, resolvers):
    """Add summations according to the Einstein convention to a term.

    In order for problems easy to be detected for users, here we just add the
    most certain Einstein summations, while give warnings when there is anything
    looking like a summation but is not added because of something suspicious.
    """

    # Strategy, find all indices to indexed bases, and replace them with
    # placeholder symbols so that we can detect other free symbols in the
    # amplitude as well.

    next_idx = [0]
    indices = []

    def replace_cb(_, *curr_indices):
        """Replace indexed quantities."""
        indices.extend(curr_indices)
        placeholder = Symbol('internalEinstPlaceholder{}'.format(next_idx[0]))
        next_idx[0] += 1
        return placeholder

    res_amp = term.amp.replace(Indexed, NonsympifiableFunc(replace_cb))
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
    for symb, use in use_tally.items():

        if use[0] != 2 and use[0] + use[1] != 2:
            # No chance to be an Einstein summation.
            continue

        if use[1] != 0:
            warnings.warn(
                'Symbol {} is not summed due to its usage in complex indices'
                    .format(symb)
            )
            continue
        if res_amp.has(symb):
            warnings.warn(
                'Symbol {} is not summed due to its usage in the amplitude'
                    .format(symb)
            )
            continue

        range_ = try_resolve_range(symb, {}, resolvers)
        if range_ is None:
            warnings.warn(
                'Symbol {} is not summed for the incapability to resolve range'
                    .format(symb)
            )
            continue

        # Now we have an Einstein summation.
        if symb not in existing_dumms:
            new_sums.append((symb, range_))
        continue

    # Make summation from Einstein convention deterministic.
    new_sums.sort(key=lambda x: (x[1].sort_key, x[0].name))

    return Term(_cat_sums(term.sums, new_sums), term.amp, term.vecs)


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

        if old in substs:
            comp_amp, new_substs = proc_delta(
                substs[old], new, sums_dict, resolvers
            )
            amp = amp * comp_amp
            if new_substs is not None:
                # The new substitution cannot involve substituted symbols.
                substs[new_substs[0]] = new_substs[1]
                # amp could now be zero.
        else:
            # Easier case, a new symbol is tried to be added.
            replace_old = {old: new}
            for i in substs.keys():
                substs[i] = substs[i].xreplace(replace_old)
            substs[old] = new

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
        return KroneckerDelta(arg1, arg2)

    eqn = Eq(arg1, arg2)

    # We try to solve for each of the dummies.  Most likely this will only be
    # executed for one loop.

    for dumm in dumms:
        range_ = sums_dict[dumm]
        sol = solve(eqn, dumm)

        if sol is S.true:
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
