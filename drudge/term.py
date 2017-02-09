"""Tensor term definition and utility."""

import abc
import collections
import functools
import itertools
import typing
from collections.abc import Iterable, Mapping, Callable, Sequence

from sympy import (
    sympify, Symbol, KroneckerDelta, DiracDelta, Eq, solve, S, Integer,
    Add, Mul, Indexed, IndexedBase, Expr, Basic, Pow)

from .canon import canon_factors
from .utils import ensure_symb, ensure_expr, sympy_key, is_higher

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
        """Initialize the symbolic range."""
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
    def bounded(self):
        """Test if the range is explicitly bounded."""
        return self._lower is not None

    @property
    def args(self):
        """Get the arguments for range creation.

        When the bounds are present, we have a triple, or we have a singleton
        tuple of only the label.
        """

        if self.bounded:
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

    @property
    def sort_key(self):
        """Get the sort key for the range."""
        key = [self._label]
        if self.bounded:
            key.extend(sympy_key(i) for i in [self._lower, self._upper])
        return key


class ATerms(abc.ABC):
    """Abstract base class for terms.

    This abstract class is meant for things that can be interpreted as a local
    collection of some tensor terms, mostly used for user input of tensor terms.

    """

    @abc.abstractproperty
    def terms(self) -> typing.Iterable['Term']:
        """Get an iterable for the terms.
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
        """Multiplies the left terms with the right terms.

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

        The possibly lazy iterable of terms will be instantiated here.
        """
        self._terms = list(terms)

    @property
    def terms(self):
        """Get the terms in the collection."""
        return self._terms


def parse_terms(obj) -> typing.Iterable['Term']:
    """Parse the object into a iterable of terms."""

    if isinstance(obj, ATerms):
        return obj.terms
    else:
        expr = ensure_expr(obj)
        return [Term((), expr, ())]


class Vec(ATerms):
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
        """Get the label for the base of the vector."""
        return self._label

    @property
    def base(self):
        """Get the base of the vector.

        This base can be subscribed to get other vectors.
        """
        return Vec(self._label, [])

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
        """Get sort key for the vector.

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

    def _sympy_(self):
        """Disable the sympification of vectors.

        This could given more sensible errors when vectors are accidentally
        attempted to be manipulated as SymPy quantities.
        """
        raise TypeError('Vectors cannot be sympified', self)

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

        Users seldom have the need to create terms directly by thi function.  So
        this constructor is mostly a developer function, no sanity checking is
        performed on the input for performance.  Most importantly, this
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
        """Get the summations of the term."""
        return self._sums

    @property
    def amp(self) -> Expr:
        """Get the amplitude expression."""
        return self._amp

    @property
    def vecs(self):
        """Gets the vectors in the term."""
        return self._vecs

    @property
    def is_scalar(self):
        """Query if the term is a scalar."""
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
        """Get the sort key for a term.

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
        """Get the singleton list of the current term.

        This property is for the rare cases where direct construction of tensor
        inputs from SymPy expressions and vectors are not sufficient.
        """
        return [self]

    def scale(self, factor):
        """Scale the term by a factor."""
        return Term(self._sums, self._amp * factor, self._vecs)

    def mul_term(self, other, dumms=None, excl=None):
        """Multiply with another tensor term."""
        lhs, rhs = self.reconcile_dumms(other, dumms, excl)
        return Term(
            lhs.sums + rhs.sums, lhs.amp * rhs.amp, lhs.vecs + rhs.vecs
        )

    def comm_term(self, other, dumms=None, excl=None):
        """Commute with another tensor term."""
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
        """Get the free symbols used in the term.
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
        """Get the factors in the amplitude expression.

        The indexed factors and factors involving dummies will be returned as a
        list, with the rest returned as a single SymPy expression.

        Error will be raised if the amplitude is not a monomial.
        """

        amp = self._amp

        if isinstance(amp, Add):
            raise ValueError('Invalid amplitude: ', amp, 'expecting monomial')
        if isinstance(amp, Mul):
            all_factors = amp.args
        else:
            all_factors = (amp,)

        dumms = {i[0] for i in self._sums}

        factors = []
        coeff = _UNITY
        for factor in all_factors:
            need_treatment = any(
                i in dumms for i in factor.atoms(Symbol)
            ) or isinstance(factor, Indexed)
            if need_treatment:
                factors.append(factor)
            else:
                coeff *= factor
            continue

        return factors, coeff

    def map(self, func, sums=None, amp=None):
        """Map the given function to the SymPy expressions in the term.

        The given function will **not** be mapped to the dummies in the
        summations.  When operations on summations are needed, an iterable
        for the new summations can be given.

        By passing the identity function, this function can also be used to
        replace the summation list or the amplitude expression.
        """

        return Term(
            self._sums if sums is None else sums,
            func(self._amp if amp is None else amp),
            tuple(i.map(func) for i in self._vecs)
        )

    def subst(self, substs, sums=None, amp=None, simultaneous=True):
        """Perform substitution on the SymPy expressions.

        This is a specialized map function, where the SymPy ``subs`` function
        will be called on each of the SymPy expression.
        """

        return self.map(
            lambda x: x.subs(substs, simultaneous=simultaneous),
            sums=sums, amp=amp
        )

    def reset_dumms(self, dumms, dummbegs=None, excl=None):
        """Reset the dummies in the term.

        The term with dummies will be returned alongside with the new dummy
        begins dictionary.  Note that the dummy begins dictionary will be
        mutated.

        ValueError will be raised when no more dummies are available.
        """

        new_sums, substs, dummbegs = self.reset_sums(
            self._sums, dumms, dummbegs, excl
        )

        return self.subst(substs, new_sums), dummbegs

    @staticmethod
    def reset_sums(sums, dumms, dummbegs=None, excl=None):
        """Reset the given summations.

        The new summation list, substitution list, and the new dummy begin
        dictionary will be returned.
        """

        if dummbegs is None:
            dummbegs = {}

        new_sums = []
        substs = []
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
            substs.append((dumm_i, new_dumm))
            dummbegs[range_i] = new_beg

            continue

        return tuple(new_sums), substs, dummbegs

    #
    # Amplitude simplification
    #

    def simplify_deltas(self, resolvers):
        """Simplify deltas in the amplitude of the expression."""

        # It is probably easy to mistakenly pass just one resolver here.
        # Special checking is done here since most mappings are also iterable.
        if isinstance(resolvers, Mapping):
            raise TypeError('Invalid range resolvers list: ', resolvers,
                            'expecting iterable')
        # Put in list to be iterated multiple times.
        resolvers = list(resolvers)

        sums_dict = dict(self._sums)
        # Here we need both fast query and remember the order.
        substs = collections.OrderedDict()
        curr_amp = self._amp

        for i in [KroneckerDelta, DiracDelta]:
            curr_amp = curr_amp.replace(i, functools.partial(
                _resolve_delta, i, sums_dict, resolvers, substs))

        # Note that here the substitutions needs to be performed in order.
        return self.subst(
            list(substs.items()),
            sums=tuple(i for i in self._sums if i[0] not in substs),
            amp=curr_amp, simultaneous=False
        )

    def simplify_amp(self, resolvers):
        """Simplify the amplitude of the term."""
        delta_proced = self.simplify_deltas(resolvers)
        return delta_proced.map(lambda x: x.simplify())

    def expand(self):
        """Expand the term into many terms."""
        expanded_amp = self.amp.expand()
        if isinstance(expanded_amp, Add):
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

        factors = []

        wrapper_base = IndexedBase('internalWrapper', shape=('internalShape',))
        amp_factors, coeff = self.amp_factors
        for i in amp_factors:
            # TODO: make it able to treat indexed inside function.
            #
            # Currently it cannot gracefully treat expressions like the
            # conjugate of an indexed quantity.
            if not isinstance(i, Indexed):
                i = wrapper_base[i]
            factors.append((
                i, (_COMMUTATIVE, sympy_key(i.base))
            ))
            continue

        for i, v in enumerate(self._vecs):
            colour = i if vec_colour is None else vec_colour(
                idx=i, vec=v, term=self
            )
            factors.append((
                v, (_NON_COMMUTATIVE, colour)
            ))
            continue

        res_sums, canoned_factors, canon_coeff = canon_factors(
            self._sums, factors, symms if symms is not None else {}
        )

        res_amp = coeff * canon_coeff
        res_vecs = []
        for i in canoned_factors:

            if isinstance(i, Vec):
                res_vecs.append(i)
            elif isinstance(i, Indexed) and i.base == wrapper_base:
                res_amp *= i.indices[0]
            else:
                res_amp *= i
            continue

        return Term(tuple(res_sums), res_amp, tuple(res_vecs))


#
# Substitution by tensor definition
# ---------------------------------
#

def subst_vec_in_term(term: Term, lhs: Vec, rhs_terms: typing.List[Term],
                      dumms, dummbegs, excl):
    """Substitute a given vector in the given term.
    """

    sums = term.sums
    vecs = term.vecs
    amp = term.amp

    for i, v in enumerate(vecs):
        if v.label == lhs.label and len(v.indices) == len(lhs.indices):
            substed_vec_idx = i
            substed_vec = v
            break
        else:
            continue
    else:
        return None  # Based on nest bind protocol.

    substs = list(zip(lhs.indices, substed_vec.indices))
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
                         dumms, dummbegs, excl):
    """Substitute a scalar factor in the term.

    Vectors is a flattened list of vectors.  The amplitude part can be a lot
    more complex.  Here we strive to replace only one instance of the lhs by two
    placeholders, the substitution is possible only if the result expands to two
    terms, each containing only one of the placeholders.
    """

    amp = term.amp

    placeholder1 = Symbol('internalSubstPlaceholder1')
    placeholder2 = Symbol('internalSubstPlaceholder2')
    found = [False]
    substs = []

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

        exts = lhs.indices
        n_exts = len(exts)

        def query_func(expr):
            """Query for a reference to a given indexed base."""
            return (
                not found[0] and isinstance(expr, Indexed)
                and expr.base.label == label
                and len(expr.indices) == n_exts
            )

        def replace_func(expr):
            """Replace the reference to the indexed base."""
            found[0] = True
            assert len(substs) == 0
            substs.extend(zip(exts, expr.indices))
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

    amp = amp.replace(Pow, decouple_pow)
    amp = amp.replace(query_func, replace_func)

    if not found[0]:
        return None

    if pow_val[0] is not None:
        amp = amp.subs(pow_placeholder, pow_val[0])
    amp = amp.simplify().expand()

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


def _prepare_subst_states(rhs_terms, substs, dumms, dummbegs, excl):
    """Prepare the substitution states.

    Here we only have partially-finished substitution state for the next loop,
    where for each substituting term on the RHS, the given external symbols in
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


def sum_term(*args, predicate=None) -> typing.List[Term]:
    """Sum the given expression.

    This method is meant for easy creation of tensor terms.  The arguments
    should start with summations and ends with the expression that is summed.

    The summations should be given as sequences, all with the first field being
    a SymPy symbol for the summation dummy.  Then comes description of the
    summation, which can be a symbolic range, SymPy expression, or iterable over
    them.

    The last argument should give the actual things to be summed, which can be
    something that can be interpreted as a collection of terms, or a callable
    that is going to return the summand when given a dictionary giving the
    action on each of the dummies.

    The predicate can be a callable going to return a boolean when called with
    same dictionary.  False values can be used the skip some terms.

    This core function is designed to be wrapped in functions working with
    full symbolic tensors.

    """

    if len(args) == 0:
        return []

    summand = args[-1]
    # Too many SymPy stuff are callable.
    if isinstance(summand, Callable) and not isinstance(summand, Basic):
        inp_terms = None
        inp_func = summand
    else:
        inp_terms = parse_terms(summand)
        inp_func = None
        if len(args) == 1:
            return list(inp_terms)

    sums, substs = _parse_sums(args[:-1])

    res = []
    for sum_i in itertools.product(*sums):
        for subst_i in itertools.product(*substs):

            # We alway assemble the call sequence here, since this part should
            # never be performance critical.
            call_seq = dict(sum_i)
            call_seq.update(subst_i)

            if not (predicate is None or predicate(call_seq)):
                continue

            if inp_terms is not None:
                curr_terms = [i.subst(
                    subst_i, sums=_cat_sums(i.sums, sum_i)
                ) for i in inp_terms]
            else:
                curr_terms = parse_terms(inp_func(call_seq))

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
# Misc public functions
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


#
# Internal functions
# ------------------
#
# Amplitude simplification
#


def _resolve_delta(form, sums_dict, resolvers, substs, *args):
    """Resolve the deltas in the given expression.

    The partial application of this function is going to be used as the
    call-back to SymPy replace function.
    """

    # We first perform the substitutions found thus far, in order.
    args = [i.subs(list(substs.items())) for i in args]
    orig = form(*args)
    dumms = [i for i in orig.atoms(Symbol) if i in sums_dict]
    if len(dumms) == 0:
        return orig

    eqn = Eq(args[0], args[1])

    # We try to solve for each of the dummies.  Most likely this will only be
    # executed for one loop.

    for dumm in dumms:
        range_ = sums_dict[dumm]
        sol = solve(eqn, dumm)

        if sol is S.true:
            # Now we can be sure that we got an identity.
            return _UNITY
        elif len(sol) > 0:
            for i in sol:
                # Try to get the range of the substituting expression.
                range_of_i = try_resolve_range(i, sums_dict, resolvers)
                if range_of_i is None:
                    continue
                if range_of_i == range_:
                    substs[dumm] = i
                    return _UNITY
                else:
                    # We assume atomic and disjoint ranges!
                    return _NAUGHT
            # We cannot resolve the range of any of the solutions.  Try next
            # dummy.
            continue
        else:
            # No solution.
            return _NAUGHT

    # When we got here, all the solutions we found have undetermined range, we
    # have to return the original form.
    return orig


#
# Canonicalization
#
# For colour of factors in a term.
#

_COMMUTATIVE = 1
_NON_COMMUTATIVE = 0
