"""Tensor term definition and utility."""

import collections
import functools
import itertools
import typing
from collections.abc import Iterable, Mapping, Callable

from sympy import (
    sympify, Symbol, KroneckerDelta, DiracDelta, Eq, solve, S, Integer,
    Add, Mul, Indexed, IndexedBase, Expr)

from .canon import canon_factors
from .utils import ensure_pair, ensure_symb, ensure_expr, sympy_key

#
# Utility constants
# -----------------
#

_UNITY = Integer(1)
_NAUGHT = Integer(0)


#
# Fundamental classes
# --------------------
#


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


class Vec:
    """Vectors.

    Vectors are the basic non-commutative quantities.  Its objects consist of an
    label for its base and some indices.  The label is allowed to be any Python
    object, although small hashable objects, like string, are advised.  The
    indices are always sympified into SymPy expressions.

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

        return ''.join([
            str(self._label), '[', ', '.join(str(i) for i in self._indices), ']'
        ])

    def __hash__(self):
        """Compute the hash value of a vector."""
        return hash((self._label, self._indices))

    def __eq__(self, other):
        """Compares the equality of two vectors."""
        return (
            (isinstance(self, type(other)) or isinstance(other, type(self))) and
            self._label == other.label and self._indices == other.indices
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
            return Term([], _UNITY, [self, other])
        else:
            return Term([], other, [self])

    def __rmul__(self, other):
        """Multiply something on the left."""

        # In principle, other should not be either a term or a vector.
        return Term([], other, [self])

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
    def amp(self) -> Expr:
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

        # TODO: optimize dummy set creation.
        dumms = set(i[0] for i in self._sums)
        frees = set(i for expr in self.exprs for i in expr.atoms(Symbol)
                    if i not in dumms)
        return frees, dumms

    @property
    def amp_factors(self):
        """Get the factors in the amplitude expression.

        The factors involving dummies will be returned as a list, with the rest
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

        dumms = {i[0] for i in self._sums}

        factors = []
        coeff = _UNITY
        for factor in all_factors:
            if any(i in dumms for i in factor.atoms(Symbol)):
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
            (i.map(func) for i in self._vecs)
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

        if dummbegs is None:
            dummbegs = {}

        new_sums = []
        substs = []
        for dumm_i, range_i in self.sums:

            # For linter.
            new_dumm = None
            new_beg = None

            beg = dummbegs[range_i] if range_i in dummbegs else 0
            for i in itertools.count(beg):

                try:
                    tentative = ensure_symb(dumms[range_i][i])
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

        return self.subst(substs, new_sums), dummbegs

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
            sums=(i for i in self._sums if i[0] not in substs),
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
            amp_terms = expanded_amp
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
            if not isinstance(i, Indexed):
                i = wrapper_base[i]
            factors.append((
                i, (_COMMUTATIVE, sympy_key(i.base))
            ))
            continue

        for i, v in enumerate(self._vecs):
            colour = i if vec_colour is None else vec_colour(idx=i, vec=v)
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

        return Term(res_sums, res_amp, res_vecs)


#
# User interface support
# ----------------------
#


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

    inp_term = _parse_term(args[-1])
    if len(args) == 1:
        return [inp_term]

    sums, substs = _parse_sums(args[:-1])

    res = []
    for sum_i in itertools.product(*sums):
        for subst_i in itertools.product(*substs):

            if predicate is not None:
                full_dict = dict(sum_i)
                full_dict.update(subst_i)
                if not predicate(full_dict):
                    continue

            res.append(inp_term.subst(
                subst_i, itertools.chain(inp_term.sums, sum_i)))

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
        return Term([], _UNITY, [term])
    else:
        return Term([], term, [])


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
                range_of_i = _try_resolve_range(i, sums_dict, resolvers)
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


def _try_resolve_range(i, sums_dict, resolvers):
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
# Canonicalization
#
# For colour of factors in a term.
#

_COMMUTATIVE = 1
_NON_COMMUTATIVE = 0
