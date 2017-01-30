"""The main drudge and tensor class definition."""

import operator
from collections.abc import Iterable

from pyspark import RDD, SparkContext

from .canonpy import Perm, Group
from .term import Range, sum_term, Term, parse_term
from .utils import ensure_symb, BCastVar


class Tensor:
    """The main tensor class.

    A tensor is an aggregate of terms distributed and managed by Spark.  Here
    most operations needed for tensors are defined.

    """

    __slots__ = [
        '_drudge',
        '_terms',
        '_local_terms'
    ]

    #
    # Term creation
    #

    def __init__(self, drudge, terms: RDD):
        """Initialize the tensor.

        This function is not designed to be called by users directly.  Tensor
        creation should be carried out by factory function inside drudges and
        the operations defined here.
        """

        self._drudge = drudge
        self._terms = terms
        self._local_terms = None

    #
    # Basic information
    #

    @property
    def terms(self):
        """Get the terms as an RDD object.

        Although for users, normally there is no need for direct manipulation of
        the terms, it is still exposed here for flexibility.
        """
        return self._terms

    @property
    def local_terms(self):
        """Gather the terms locally into a list.

        The list returned by this is for read-only and should **never** be
        mutated.
        """
        if self._local_terms is None:
            self._local_terms = self._terms.collect()
        return self._local_terms

    @property
    def n_terms(self):
        """Get the number of terms.

        A zero number of terms signatures a zero tensor.
        """
        if self._local_terms is not None:
            return len(self._local_terms)
        else:
            return self._terms.count()

    def cache(self):
        """Cache the terms in the tensor.

        This method should be called when this tensor is an intermediate result
        that is used multiple times.
        """
        self._terms.cache()
        return

    #
    # Small manipulations
    #

    def apply(self, func):
        """Apply the given function to the RDD of terms.

        Since this method is mostly a developer function, sanity check is not
        performed.  But note that the given function will be called with the RDD
        of the terms in the current tensor, and another RDD should be returned.
        """
        return Tensor(self._drudge, func(self._terms))

    #
    # Here for a lot of methods, we have two versions, with one being public,
    # another being private with a leading underscore.  The private version
    # operates on given RDD of terms and returns another RDD of terms.  The
    # public version operates on the terms of the current tensor, and return
    # another tensor.
    #

    @property
    def free_vars(self):
        """The free variables in the tensor."""
        return self._free_vars(self._terms)

    @staticmethod
    def _free_vars(terms):
        return terms.map(lambda term: term.symbs[0]).reduce(operator.or_)

    def reset_dumms(self):
        """Reset the dummies.

        The dummies will be set to the canonical dummies according to the order
        in the summation list.  This method is especially useful on
        canonicalized tensors.
        """
        return self.apply(self._reset_dumms)

    def _reset_dumms(self, terms, excl=None):
        """Get terms with dummies reset."""

        free_vars = self._free_vars(terms)
        if excl is None:
            excl = set()
        excl |= free_vars

        dumms = self._drudge.dumms
        res_terms = terms.map(
            lambda term: term.reset_dumms(dumms=dumms.value, excl=excl)[0]
        )
        return res_terms

    def simplify_amps(self):
        """Simplify the amplitudes in the tensor.

        This method simplifies the amplitude in the tensor of the tensor, by
        using the facility from SymPy and tensor specific facilities for deltas.
        The zero terms will be filtered out as well.
        """
        return self.apply(self._simplify_amps)

    def _simplify_amps(self, terms):
        """Get the terms with amplitude simplified."""
        resolvers = self._drudge.resolvers
        simplified_terms = terms.map(
            lambda term: term.simplify_amp(resolvers=resolvers.value)
        ).filter(lambda term: term.amp != 0)
        return simplified_terms

    def expand(self):
        """Expand the terms in the tensor.

        By calling this method, terms in the tensor whose amplitude is the
        addition of multiple parts will be expanded into multiple terms.
        """
        return self.apply(self._expand)

    def sort(self):
        """Sort the terms in the tensor.

        The terms will generally be sorted according to increasing complexity.

        """
        self.apply(self._sort)

    @staticmethod
    def _sort(terms: RDD):
        """Compute the terms in the tensor."""
        return terms.sortBy(lambda term: term.sort_key)

    @staticmethod
    def _expand(terms):
        """Get terms after they are fully expanded."""
        return terms.flatMap(lambda term: term.expand())

    def merge(self):
        """Merge terms with the same vector and summation part.

        This function merges terms only when their summation list and vector
        part are *syntactically* the same.  So it is more useful when the
        canonicalization has been performed and the dummies reset.
        """
        return self.apply(self._merge)

    @staticmethod
    def _merge(terms):
        """Get the term when they are attempted to be merged."""
        return terms.map(
            lambda term: ((term.sums, term.vecs), term.amp)
        ).reduceByKey(operator.add).map(
            lambda x: Term(x[0][0], x[1], x[0][1])
        )

    #
    # Canonicalization
    #

    def canon(self):
        """Canonicalize the terms in the tensor.

        This method will first expand the terms in the tensor.  Then the
        canonicalization algorithm is going to be applied to each of the terms.
        Note that this method does not rename the dummies.
        """
        return self.apply(self._canon)

    def _canon(self, terms):
        """Compute the canonicalized terms."""

        symms = self._drudge.symms

        vec_colour = self._drudge.vec_colour
        # Normally a static function, not broadcast variable.

        expanded_terms = self._expand(terms)
        canoned = expanded_terms.map(
            lambda term: term.canon(symms=symms.value, vec_colour=vec_colour)
        )
        return canoned

    def normal_order(self):
        """Normal order the terms in the tensor.

        The actual work is dispatched to the drudge, who has domain specific
        knowledge about the noncommutativity of the vectors.
        """
        return self.apply(self._drudge.normal_order)

    #
    # The driver simplification.
    #

    def simplify(self):
        """Simplify the tensor.

        This is the master driver function for tensor simplification.
        """
        return self.apply(self._simplify)

    def _simplify(self, terms):
        """Get the terms in the simplified form."""

        # First we make the vector part normal-ordered.
        terms = self._drudge.normal_order(terms)

        # Simplify things like zero or deltas.
        terms = self._simplify_amps(terms)

        # Canonicalize the terms and see if they can be merged.
        terms = self._canon(terms)
        terms = self._reset_dumms(terms)
        terms = self._sort(terms)
        terms = self._merge(terms)

        # Finally simplify the merged amplitude again.
        terms = self._simplify_amps(terms)

        return terms

    #
    # Comparison operations
    #

    def __eq__(self, other):
        """Compare the equality of tensors.

        Note that this function only compares the syntactical equality of
        tensors.  Mathematically equal tensors might be compared to be unequal
        by this function when they are not simplified.

        Note that this function gathers all terms in the tensor and can be very
        expensive.  So it is mostly suitable for testing and debugging on small
        problems only.
        """

        n_terms = self.n_terms
        if isinstance(other, Tensor):
            return self.local_terms == other.local_terms
        elif other == 0:
            return n_terms == 0
        else:
            if n_terms != 1:
                return False
            else:
                term = self.local_terms[0]
                return term == parse_term(other)

        assert False

    #
    # Mathematical operations
    #

    _op_priority = 20.0

    def __add__(self, other):
        """Add the two tensors together.

        The terms in the two tensors will be concatenated together, without any
        further processing.
        """

        if not isinstance(other, Tensor):
            other = self._drudge.sum(other)
        return Tensor(self._drudge, self._terms.union(other.terms))


class Drudge:
    """The main drudge class.

    A drudge is a robot who can help you with the menial tasks of symbolic
    manipulation for tensorial and noncommutative alegbras.  Due to the
    diversity and non-uniformity of tensor and noncommutative algebraic
    problems, to set up a drudge, information about the problem needs to be
    given.  Here this is a base class, where the basic operations are defined.
    Different problems could subclass this base class with customized behaviour.

    """

    # We do not need slots here.  There is generally only one drudge instance.

    def __init__(self, ctx: SparkContext):
        """Initialize the drudge.
        """

        self._ctx = ctx

        self._dumms = BCastVar(self._ctx, {})
        self._symms = BCastVar(self._ctx, {})
        self._resolvers = BCastVar(self._ctx, [])

    #
    # General properties
    #
    # Subclasses normally just need to use the methods in this section to add
    # some additional information.  The method here generally does not need to
    # be overridden.
    #

    def set_dumms(self, range_: Range, dumms):
        """Set the dummies for a range.

        Note that this function overwrites the existing dummies if the range has
        already been given.
        """
        self._dumms.var[range_] = [ensure_symb(i) for i in dumms]

    @property
    def dumms(self):
        """Get the broadcast form of the dummies dictionary.
        """
        return self._dumms.bcast

    def set_symm(self, base, *symms):
        """Get the symmetry for a given base.

        Permutation objects in the arguments are interpreted as single
        generators, other values will be attempted to be iterated over to get
        their entries, which should all be permutations.
        """

        gens = []
        for i in symms:
            if isinstance(i, Perm):
                gens.append(i)
            elif isinstance(i, Iterable):
                gens.extend(i)
            else:
                raise TypeError('Invalid generator: ', i,
                                'expecting Perm or iterable of Perms')
            continue

        self._symms.var[base] = Group(gens)

    @property
    def symms(self):
        """Get the broadcast form of the symmetries."""
        return self._symms.bcast

    def add_resolver(self, resolver):
        """Append a resolver to the list of resolvers.

        The given resolver can be either a mapping from SymPy expression,
        including atomic symbols, to the corresponding ranges.  Or a callable to
        be called with SymPy expressions.  For callable resolvers, None can be
        returned to signal the incapability to resolve the expression.  Then the
        resolution will be dispatched to the next resolver.
        """
        self._resolvers.var.append(resolver)
        return

    def add_resolver_for_dumms(self):
        """Add the resolver for the dummies for each range.

        With this method, the default dummies for each range will be resolved to
        be within the range for all of them.  This method should normally be
        called by all subclasses.

        Note that dummies added later will not be automatically added.  This
        method can be called again.
        """

        dumm_resolver = {}
        for k, v in self._dumms.ro.items():
            for i in v:
                dumm_resolver[i] = k
                continue
            continue
        self.add_resolver(dumm_resolver)

    @property
    def resolvers(self):
        """Get the broadcast form of the resolvers."""
        return self._resolvers.bcast

    #
    # Vector-related properties
    #
    # The methods here is highly likely to be overridden for different
    # non-commutative algebraic systems in different problems.
    #

    @property
    def vec_colour(self):
        """Get the vector colour function.

        Note that this accessor accesses the **function**, rather than directly
        computes the colour for any vector.
        """
        return None

    @staticmethod
    def normal_order(terms):
        """Normal order the terms in the given tensor.

        This method should be called with the RDD of some terms, and another RDD
        of terms, where all the vector parts are normal ordered according to
        domain-specific rules, should be returned.

        By default, we work for the free algebra.  So nothing is done by this
        function.  For noncommutative algebraic system, this function needs to
        be overridden to return an RDD for the normal-ordered terms from the
        given terms.
        """

        return terms

    #
    # Tensor creation
    #

    def sum(self, *args) -> Tensor:
        """Create a tensor for the given summation.

        This is the core function for creating tensors from scratch.  The last
        argument will be interpreted as the quantity that is summed over.
        Terms, vectors, or SymPy expressions are supported.  Earlier argument,
        if there is any, should be dummy/range pairs giving the symbolic
        summations to be carried out.
        """
        return Tensor(
            self, self._ctx.parallelize(sum_term(*args))
        )
