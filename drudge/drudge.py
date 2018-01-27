"""The main drudge and tensor class definition."""

import collections
import contextlib
import functools
import inspect
import operator
import pickle
import sys
import types
import typing
import warnings
from collections.abc import Iterable, Sequence

from IPython.display import Math, display
from pyspark import RDD, SparkContext
from sympy import (
    IndexedBase, Symbol, Indexed, Wild, symbols, sympify, Expr, Add, Sum, Matrix
)
from sympy.concrete.summations import eval_sum_symbolic

from .canonpy import Perm, Group
from .drs import compile_drs, DrsEnv, DrsSymbol
from .report import Report, ScalarLatexPrinter
from .term import (
    Range, sum_term, Term, Vec, subst_factor_term, subst_vec_term, parse_terms,
    einst_term, diff_term, try_resolve_range, rewrite_term, Sum_expander,
    expand_sums_term, ATerms, simplify_amp_sums_term
)
from .utils import (
    ensure_symb, BCastVar, nest_bind, prod_, sympy_key, SymbResolver
)


class Tensor:
    """The main tensor class.

    A tensor is an aggregate of terms distributed and managed by Spark.  Here
    most operations needed for tensors are defined.

    Normally, tensor instances are created from drudge methods or tensor
    operations.  Direct invocation of its constructor is seldom in user scripts.

    """

    __slots__ = [
        '_drudge',
        '_terms',
        '_local_terms',
        '_free_vars',
        '_expanded',
        '_repartitioned'
    ]

    #
    # Term creation
    #

    def __init__(self, drudge: 'Drudge', terms: RDD,
                 free_vars: typing.Set[Symbol] = None,
                 expanded=False, repartitioned=False):
        """Initialize the tensor.

        This function is not designed to be called by users directly.  Tensor
        creation should be carried out by factory function inside drudges and
        the operations defined here.

        The default values for the keyword arguments are always the safest
        choice, for better performance, manipulations are encouraged to have
        proper consideration of all the keyword arguments.
        """

        self._drudge = drudge
        self._terms = terms

        self._local_terms = None  # type: typing.List[Term]

        self._free_vars = free_vars
        self._expanded = expanded
        self._repartitioned = repartitioned

    # To be used by the apply method.
    _INIT_ARGS = {
        'free_vars': '_free_vars',
        'expanded': '_expanded',
        'repartitioned': '_repartitioned'
    }

    #
    # Basic information
    #

    @property
    def drudge(self):
        """The drudge created the tensor."""
        return self._drudge

    @property
    def terms(self):
        """The terms in the tensor, as an RDD object.

        Although for users, normally there is no need for direct manipulation of
        the terms, it is still exposed here for flexibility.
        """
        return self._terms

    @property
    def local_terms(self) -> typing.List[Term]:
        """Gather the terms locally into a list.

        The list returned by this is for read-only and should **never** be
        mutated.

        .. warning::

            This method will gather all terms into the memory of the driver.

        """
        if self._local_terms is None:
            self._local_terms = self._terms.collect()
        else:
            pass

        return self._local_terms

    @property
    def n_terms(self) -> int:
        """Get the number of terms.

        A zero number of terms signatures a zero tensor.  Accessing this
        property will make the tensor to be cached automatically.
        """
        if self._local_terms is not None:
            return len(self._local_terms)
        else:
            self.cache()  # We never get a tensor just to count its terms.
            return self._terms.count()

    def cache(self):
        """Cache the terms in the tensor.

        This method should be called when this tensor is an intermediate result
        that will be used multiple times.  The tensor itself will be returned
        for the ease of chaining.
        """

        self._terms.cache()
        return self

    def repartition(self, num_partitions=None, cache=False):
        """Repartition the terms across the Spark cluster.

        This function should be called when the terms need to be rebalanced
        among the workers.  Note that this incurs an Spark RDD shuffle operation
        and might be very expensive.  Its invocation and the number of
        partitions used need to be fine-tuned for different problems to achieve
        good performance.

        Parameters
        ----------

        num_partitions : int
            The number of partitions.  By default, the number is read from the
            drudge object.

        cache : bool
            If the result is going to be cached.

        """

        if not self._repartitioned:

            num_partitions = (
                self._drudge.num_partitions if num_partitions is None else
                num_partitions
            )
            if num_partitions is None:
                raise ValueError('No default number of partitions available')

            self._terms = self._terms.repartition(num_partitions)
            self._repartitioned = True

        if cache:
            self.cache()
        return self

    @property
    def is_scalar(self):
        """If the tensor is a scalar.

        A tensor is considered a scalar when none of its terms has a vector
        part.  This property will make the tensor automatically cached.
        """

        self.cache()

        # Work around a pyspark bug by doing the reduction locally.
        return all(
            self._terms.map(lambda x: x.is_scalar).collect()
        )

    @property
    def free_vars(self) -> typing.Set[Symbol]:
        """The free variables in the tensor.
        """
        if self._free_vars is None:
            self._free_vars = self._get_free_vars(self._terms)

        return self._free_vars

    @staticmethod
    def _get_free_vars(terms) -> typing.Set[Symbol]:
        """Get the free variables in the given terms."""

        # The terms are definitely going to be used for other purposes.
        terms.cache()

        return terms.map(
            lambda term: term.free_vars
        ).aggregate(set(), _union, _union)
        # TODO: investigate performance characteristic with treeAggregate.

    @property
    def expanded(self):
        """If the tensor is already expanded."""
        return self._expanded

    @property
    def repartitioned(self):
        """If the terms in the tensor is already repartitioned."""
        return self._repartitioned

    def has_base(self, base: typing.Union[IndexedBase, Symbol, Vec]) -> bool:
        """Find if the tensor has the given scalar or vector base.

        Parameters
        ----------

        base
            The base whose presence is to be queried.  When it is indexed base
            or a plain symbol, its presence in the amplitude part is tested.
            When it is a vector, its presence in the vector part is tested.

        """

        # A tensor is barely needed by a has_base decision only.
        self.cache()

        # Work around a possible pyspark bug in reduce.
        return any(
            self._terms.map(functools.partial(Term.has_base, base=base))
                .collect()
        )

    #
    # Printing support
    #

    def __repr__(self):
        """Get the machine string representation.

        In normal execution environment, only the memory address is displayed,
        since the tensor may or may not be evaluated yet.  In drudge scripts,
        the readable string representation is returned.
        """

        if self._drudge.inside_drs:
            return str(self)
        else:
            return super().__repr__()

    def __str__(self):
        """Get the string representation of the tensor.

        Note that this function will **gather** all terms into the driver.

        """
        if self.n_terms == 0:
            return '0'
        else:
            return '\n + '.join(str(i) for i in self.local_terms)

    def latex(self, **kwargs):
        r"""Get the latex form for the tensor.

        The actual printing is dispatched to the drudge object for the
        convenience of tuning the appearance.

        All keyword arguments are forwarded to the
        :py:meth:`Drudge.format_latex` method.

        """
        return self._drudge.format_latex(self, **kwargs)

    def display(self, if_return=True, **kwargs):
        """Display the tensor in interactive IPython notebook sessions.

        Parameters
        ----------

        if_return
            If the resulted equation be returned rather than directly displayed.
            It can be disabled for displaying equation in the middle of a
            Jupyter cell.

        kwargs
            All the rest of the keyword arguments are forwarded to the
            :py:meth:`Drudge.format_latex` method.

        """

        form = Math(self.latex(**kwargs))
        if if_return:
            return form
        else:
            display(form)
            return

    #
    # Pickling support
    #

    def __getstate__(self):
        """Get the current state of the tensor.

        Here we just have the local terms.  Other cached information are
        discarded.
        """
        return self.local_terms

    def __setstate__(self, state):
        """Set the state for the new tensor.

        This function reads the drudge to use from the module attribute, which
        is set in the :py:meth:`Drudge.pickle_env` method.
        """

        global current_drudge
        drudge = current_drudge
        if drudge is None:
            raise ValueError(
                'Tensor objects cannot be unpickled, '
                'need to be inside Drudge.pickle_env'
            )

        assert isinstance(drudge, Drudge)

        # Subclasses might be of __init__ function of a different signature.
        Tensor.__init__(self, drudge, drudge.ctx.parallelize(state))
        return

    #
    # Small manipulations
    #

    def apply(self, func, **kwargs):
        """Apply the given function to the RDD of terms.

        This function is analogous to the replace function of Python named
        tuples, the same value from self for the tensor initializer is going to
        be used when it is not given.  The terms get special treatment since it
        is the centre of tensor objects.  The drudge is kept the same always.

        Users generally do not need this method.  It is exposed here just for
        flexibility and convenience.

        .. warning::

            For developers:  Note that the resulted tensor will inherit all
            unspecified keyword arguments from self.  This method can give
            *unexpected results* if certain arguments are not correctly reset
            when they need to.  For instance, when expanded is not reset when
            the result is no longer guaranteed to be in expanded form, later
            expansions could be skipped when they actually need to be performed.

            So all functions using this methods need to be reviewed when new
            property are added to tensor class.  Direct invocation of the tensor
            constructor is a much safe alternative.

        """

        for k, v in self._INIT_ARGS.items():
            if k not in kwargs:
                kwargs[k] = getattr(self, v)

        return Tensor(self._drudge, func(self._terms), **kwargs)

    #
    # Here for a lot of methods, we have two versions, with one being public,
    # another being private with a leading underscore.  The private version
    # operates on given RDD of terms and returns another RDD of terms.  The
    # public version operates on the terms of the current tensor, and return
    # another tensor.
    #

    def reset_dumms(self, excl=None):
        """Reset the dummies.

        The dummies will be set to the canonical dummies according to the order
        in the summation list.  This method is especially useful on
        canonicalized tensors.

        Parameters
        ----------

        excl
            A set of symbols to be excluded in the dummy selection.  This option
            can be useful when some symbols already used as dummies are planned
            to be used for other purposes.

        """

        free_vars = self.free_vars
        if excl is None:
            excl = free_vars
        else:
            excl |= free_vars

        return self.apply(functools.partial(self._reset_dumms, excl=excl))

    def _reset_dumms(self, terms: RDD, excl) -> RDD:
        """Get terms with dummies reset.

        Note that this function does not automatically add the free variables in
        the terms to the excluded symbols.
        """

        dumms = self._drudge.dumms
        res_terms = terms.map(
            lambda term: term.reset_dumms(dumms=dumms.value, excl=excl)[0]
        )
        return res_terms

    def simplify_amps(self):
        """Simplify the amplitudes in the tensor.

        This method simplifies the amplitude in the terms of the tensor by using
        the facility from SymPy.  The zero terms will be filtered out as well.

        """

        # Some free variables might be canceled.
        return self.apply(
            self._simplify_amps, free_vars=None, repartitioned=False
        )

    @staticmethod
    def _simplify_amps(terms):
        """Get the terms with amplitude simplified by SymPy."""

        simplified_terms = terms.map(
            lambda term: term.map(lambda x: x.simplify(), skip_vecs=True)
        ).filter(_is_nonzero)

        return simplified_terms

    def simplify_deltas(self):
        """Simplify the deltas in the tensor.

        Kronecker deltas whose operands contains dummies will be attempted to be
        simplified.
        """

        return Tensor(
            self._drudge,
            self._simplify_deltas(self._terms, expanded=self._expanded),
            expanded=True,
        )

    def _simplify_deltas(self, terms, expanded):
        """Simplify the deltas in the terms.

        This function will leave the resulted terms in expanded form.
        """

        if not expanded:
            terms = self._expand(terms)

        resolvers = self._drudge.resolvers

        return terms.map(
            lambda x: x.simplify_deltas(resolvers.value)
        ).filter(_is_nonzero)

    def simplify_sums(self, excl_bases=True, aggr=True, simplify=None):
        """Simplify the summations within the amplitude in the tensor.

        This method attempts to perform simplification for concrete summations
        over dummies that does not appear on the vector part at all.

        First, bounded summations with dummies not actually involved in the term
        will be replaced by a multiplication with its size.  Then simplification
        is going to be performed on other summations within the amplitude part.

        Parameters
        ----------

        excl_bases
            If summations involved by indexed bases are also going to be
            excluded from attempts of simplifications.  This can be safely set
            to true unless customized rules exists.

        aggr
            If summations involved by the same set of factors are going to be
            attempted together.  Or all subset of the summations are going to be
            tested in turn.

        simplify
            The call-back to simplify the internally summed factors of the
            amplitude.  It will normally be called with SymPy ``Sum`` objects.
            If the same expression or ``None`` is returned, it will be
            considered to be not simplifiable any more.  By default, the
            :py:meth:`simplify_amp_sum` attribute of the current drudge object
            will be used.  Note that this attribute needs to be serializable in
            Spark environment.  Notably they cannot be a non-static method of
            the drudge object.
        """

        return Tensor(self._drudge, self._simplify_sums(
            self._terms, excl_bases=excl_bases, aggr=aggr, simplify=simplify
        ))

    def _simplify_sums(
            self, terms: RDD, excl_bases=True, aggr=True, simplify=None
    ):
        """Simplify the summations in the given terms."""

        if simplify is None:
            simplify = self._drudge.simplify_amp_sum

        # Make it a two-step process for future extensibility.
        terms = terms.map(lambda x: x.simplify_trivial_sums())
        terms = terms.map(functools.partial(
            simplify_amp_sums_term, excl_bases=excl_bases, aggr=aggr,
            simplify=simplify
        ))
        return terms

    def expand(self):
        """Expand the terms in the tensor.

        By calling this method, terms in the tensor whose amplitude is the
        addition of multiple parts will be expanded into multiple terms.
        """
        if self._expanded:
            return self
        else:
            return self.apply(self._expand, expanded=True, repartitioned=False)

    @staticmethod
    def _expand(terms):
        """Get terms after they are fully expanded."""
        return terms.flatMap(lambda term: term.expand())

    def shallow_expand(self):
        """Expand terms with addition amplitudes into multiple terms.

        Different from the :py:meth:`expand` method, this method only expands
        a term when its amplitude is written as an addition on the
        top-level.  This is useful for cases where the inner structures of the
        amplitude expression tree is intended to be preserved.
        """
        return self.bind(self._shallow_expand)

    @staticmethod
    def _shallow_expand(term: Term) -> typing.Iterable[Term]:
        """Expand a term shallowly."""
        if isinstance(term.amp, Add):
            return [
                Term(sums=term.sums, amp=i, vecs=term.vecs)
                for i in term.amp.args
            ]
        else:
            return [term]

    def sort(self):
        """Sort the terms in the tensor.

        The terms will generally be sorted according to increasing complexity.
        """
        return self.apply(self._sort)

    @staticmethod
    def _sort(terms: RDD):
        """Sort the terms in the tensor."""
        return terms.sortBy(lambda term: term.sort_key)

    def merge(self, consts=None, gens=None):
        """Merge some terms.

        This function attempts to merge some terms into smaller number of terms.
        Two terms can be merged only when their summation list, vector part, and
        selected factors are *syntactically* the same.  So it is more useful
        when the canonicalization has been performed and the dummies reset.

        When simple merge is disabled, nothing is put into the selected factors,
        all terms with the same summations and vectors are tried to be merged.
        When simple merge is enabled, inside the amplitude, any factor
        containing one of the dummy variables and all indexed quantities are put
        into the selected factors.  By default, all other factors are considered
        to be symbols standing for some constants and will be merged.  If the
        ``consts`` argument is given, only the symbols given there will be
        considered to be constant symbols for coefficients, with all factors
        with any other symbol put into the selected factors.  If the ``gens``
        argument is given, only symbols given here are considered to be
        generators of the polynomial, with all others considered to be
        constants.

        """

        # All the traits could be invalidated by merging.
        return Tensor(
            self._drudge, self._merge(self._terms, consts, gens)
        )

    def _merge(self, terms, consts, gens):
        """Get the term when they are attempted to be merged."""
        if not self._drudge.simple_merge and consts is None and gens is None:
            specials = None
        else:
            specials = _DecomposeSpecials(consts, gens)

        return terms.map(
            functools.partial(_decompose_term, specials=specials)
        ).reduceByKey(operator.add).map(_recover_term)

    #
    # Canonicalization
    #

    def canon(self):
        """Canonicalize the terms in the tensor.

        This method will first expand the terms in the tensor.  Then the
        canonicalization algorithm is going to be applied to each of the terms.
        Note that this method does not rename the dummies.
        """
        return self.apply(
            functools.partial(self._canon, expanded=self._expanded),
            expanded=True, repartitioned=self._expanded and self._repartitioned
        )

    def _canon(self, terms, expanded):
        """Compute the canonicalized terms."""

        symms = self._drudge.symms

        vec_colour = self._drudge.vec_colour
        # Normally a static function, not broadcast variable.

        if not expanded:
            expanded_terms = self._expand(terms)
        else:
            expanded_terms = terms
        canoned = expanded_terms.map(
            lambda term: term.canon(symms=symms.value, vec_colour=vec_colour)
        )
        return canoned

    def normal_order(self):
        """Normal order the terms in the tensor.

        The actual work is dispatched to the drudge, who has domain specific
        knowledge about the noncommutativity of the vectors.

        """

        # Free variables, expanded, and repartitioned can all be invalidated.
        return Tensor(
            self._drudge, self._drudge.normal_order(self.terms)
        )

    #
    # The driver simplification.
    #

    def simplify(self):
        """Simplify the tensor.

        This is the master driver function for tensor simplification.  Inside
        drudge scripts, it also make eager evaluation and repartition the terms
        among the Spark workers, with the result cached.  This is for the ease
        of users unfamiliar with the Spark lazy execution model.

        """

        result = Tensor(
            self._drudge, self._simplify(self._terms), expanded=True
        )

        if self._drudge.inside_drs:
            result.repartition(cache=True)
            _ = result.n_terms

        return result

    def _simplify(self, terms):
        """Get the terms in the simplified form."""

        num_partitions = self._drudge.num_partitions

        repartitioned = self._repartitioned
        if not self._expanded:
            terms = self._expand(terms)
            repartitioned = False

        if not repartitioned and num_partitions is not None:
            terms = terms.repartition(num_partitions)

        # First we make the vector part normal-ordered.
        terms = self._drudge.normal_order(terms)
        if num_partitions is not None:
            terms = terms.repartition(num_partitions)

        # Simplify the amplitude part.
        terms = self._simplify_amps(terms)
        terms = self._simplify_sums(terms)
        terms = self._simplify_amps(terms)
        terms = self._simplify_deltas(terms, False)

        # Canonicalize the terms and see if they can be merged.
        terms = self._canon(terms, True)
        # In rare cases, normal order could make the result unexpanded.
        #
        # TODO: Find a design to skip repartition in most cases.

        free_vars = self._get_free_vars(terms)
        terms = self._reset_dumms(terms, excl=free_vars)
        terms = self._merge(terms, None, None)

        # Finally simplify the merged amplitude again.
        if self._drudge.full_simplify:
            terms = self._simplify_amps(terms)

        # Make the final expansion.
        terms = self._expand(terms)

        return terms

    #
    # Comparison operations
    #

    def __eq__(self, other):
        """Compare the equality of tensors.

        Note that this function only compares the syntactical equality of
        tensors.  Mathematically equal tensors might be compared to be unequal
        by this function when they are not simplified.

        Note that only comparison with zero is performed by counting the number
        of terms distributed.  Or this function gathers all terms in both
        tensors and can be very expensive.  So direct comparison of two tensors
        is mostly suitable for testing and debugging on small problems only.
        For large scale problems, it is advised to compare the simplified
        difference with zero.
        """

        n_terms = self.n_terms
        if isinstance(other, Tensor):
            return set(self.local_terms) == set(other.local_terms)
        elif other == 0:
            return n_terms == 0
        else:
            return self._drudge.sum(other)

    #
    # Mathematical operations
    #

    _op_priority = 20.0

    def __add__(self, other):
        """Add the two tensors together.

        The terms in the two tensors will be concatenated together, without any
        further processing.

        In addition to full tensors, tensor inputs can also be directly added.
        """
        return self._add(other)

    def __radd__(self, other):
        """Add tensor with something in front."""
        return self._add(other)

    def _add(self, other):
        """Add tensor with another thing."""
        if not isinstance(other, Tensor):
            other = self._drudge.sum(other)

        if self._free_vars is not None and other._free_vars is not None:
            free_vars = self.free_vars | other.free_vars
        else:
            free_vars = None

        return Tensor(
            self._drudge, self._terms.union(other.terms),
            free_vars=free_vars,
            expanded=self._expanded and other.expanded
        )

    def __sub__(self, other):
        """Subtract another tensor from this tensor.
        """
        return self._add(-other)

    def __rsub__(self, other):
        """Subtract the tensor from another quantity."""
        return (-self)._add(other)

    def __neg__(self):
        """Negate the current tensor.

        The result will be equivalent to multiplication with :math:`-1`.
        """
        return self.apply(
            lambda terms: terms.map(lambda x: x.scale(-1))
        )

    def __mul__(self, other) -> 'Tensor':
        """Multiply the tensor.

        This multiplication operation is done completely within the framework of
        free algebras.  The vectors are only concatenated without further
        processing.  The actual handling of the commutativity should be carried
        out at the normal ordering operation for different problems.

        In addition to full tensors, tensors can also be multiplied to user
        tensor input directly.
        """
        return self._mul(other)

    def __rmul__(self, other):
        """Multiply the tensor on the right.
        """
        return self._mul(other, right=True)

    def _mul(self, other, right=False):
        """Multiply the tensor with another."""
        prod, free_vars, expanded = self._cartesian_terms(other, right)

        dumms = self._drudge.dumms
        return Tensor(self._drudge, prod.map(
            lambda x: x[0].mul_term(x[1], dumms=dumms.value, excl=free_vars)
        ), free_vars=free_vars, expanded=expanded)

    def __or__(self, other):
        """Compute the commutator with another tensor.

        In the same way as multiplication, this can be used for both full
        tensors and local tensor input.
        """
        return self._comm(other)

    def __ror__(self, other):
        """Compute the commutator with another tensor on the right."""
        return self._comm(other, right=True)

    def _comm(self, other, right=False):
        """Compute the commutator."""

        if isinstance(other, Expr):
            msg = 'Taking commutator with commutative expression `{}`'.format(
                other
            )
            warnings.warn(msg)

        prod, free_vars, expanded = self._cartesian_terms(other, right)

        dumms = self._drudge.dumms
        return Tensor(self._drudge, prod.flatMap(
            lambda x: x[0].comm_term(x[1], dumms=dumms.value, excl=free_vars)
        ), free_vars=free_vars, expanded=expanded)

    def _cartesian_terms(self, other, right):
        """Cartesian the terms with the terms in another tensor.

        The other tensor will be attempted to be interpreted as a tensor when it
        is not given as one.  And the free variables used in both tensors will
        also be returned since it is going to be used frequently.
        """

        if isinstance(other, Tensor):

            if right:
                prod = other.terms.cartesian(self._terms)
            else:
                prod = self._terms.cartesian(other.terms)

            free_vars = self.free_vars | other.free_vars
            expanded = self._expanded and other._expanded

        else:
            # Special optimized version when the other terms are local.

            other_terms = parse_terms(other)
            if len(other_terms) > 1:
                prod = self._terms.flatMap(lambda term: [
                    (i, term) if right else (term, i)
                    for i in other_terms
                ])
            else:
                # Special optimization when we just have one term.
                other_term = other_terms[0]
                prod = self._terms.map(
                    lambda term:
                    (other_term, term) if right else (term, other_term)
                )

            free_vars = set.union(*[
                i.free_vars for i in other_terms
            ])
            free_vars |= self.free_vars
            expanded = False

        return prod, free_vars, expanded

    def __truediv__(self, other):
        """Divide tensor by a scalar quantity."""

        other = sympify(other)
        return self.apply(
            lambda terms: terms.map(lambda x: x.scale(1 / other)),
            free_vars=None
        )

    def __rtruediv__(self, other):
        """Make division over a tensor."""
        raise NotImplementedError('General tensors cannot be divided over.')

    #
    # Substitution
    #

    def subst(
            self, lhs, rhs, wilds=None, full_balance=False, excl=None,
            simult=True
    ):
        """Substitute the all appearance of the defined tensor.

        When the given LHS is a plain SymPy symbol, all its appearances in the
        amplitude of the tensor will be replaced.  Or the LHS can also be
        indexed SymPy expression or indexed Vector, for which all of the
        appearances of the indexed base or vector base will be attempted to be
        matched against the indices on the LHS.  When a matching succeeds for
        all the indices, the RHS, with the substitution found in the matching
        performed, will be replace the indexed base in the amplitude, or the
        vector.  Note that for scalar LHS, the RHS must contain no vector.  For
        vector LHS, also supported is a product of vectors.  This will lead to
        pattern matching inside the vector part.

        A special case is when the LHS is a plain integral unity.  In this case,
        the scalar terms will be replaced by product with the RHS, as if the
        identity element has been replaced by the RHS.

        Parameters
        ----------

        lhs
            The left-hand side of the substitution.

        rhs
            The right-hand side of the substituion.

        wilds
            Since we do not commonly define tensors with wild symbols, an option
            ``wilds`` can be used to give a mapping translating plain symbols on
            the LHS and the RHS to the wild symbols that would like to be used.
            The default value of None could make all **plain** symbols in the
            indices of the LHS to be translated into a wild symbol with the same
            name and no exclusion. And empty dictionary can be used to disable
            all such automatic translation.  The default value of None should
            satisfy most needs.

        full_balance
            If load-balancing is going to be performed after each substitution
            step.  This is normally not needed.

        excl
            Symbols that needs to be excluded to be used as dummies.

        simult
            The core substitution algorithm works by making one substitution at
            a time until no more left-hand side can be matched successfully.  In
            this way, if the pattern in the LHS reappears on the RHS, infinite
            loop will be resulted.  When it is set to true, the
            newly-substituted part will no longer participate in the matching.

            By default, it is true.  Internally, it decorates the bases on the
            LHS to special forms, so that the originally-present ones are
            different from the newly-substituted ones.  Note that its disabling
            makes sense only when the LHS is a complex pattern.

        Examples
        --------

        For instance, we can have a very simple tensor, the outer product of the
        same vector,

        .. doctest::

            >>> dr = Drudge(SparkContext())
            >>> r = Range('R')
            >>> a, b = dr.set_dumms(r, symbols('a b c d e f'))[:2]
            >>> dr.add_default_resolver(r)
            >>> x = IndexedBase('x')
            >>> v = Vec('v')
            >>> tensor = dr.einst(x[a] * x[b] * v[a] * v[b])
            >>> str(tensor)
            'sum_{a, b} x[a]*x[b] * v[a] * v[b]'

        We can replace the indexed base by the product of a matrix with another
        indexed base,

        .. doctest::

            >>> o = IndexedBase('o')
            >>> y = IndexedBase('y')
            >>> res = tensor.subst(x[a], dr.einst(o[a, b] * y[b]))
            >>> str(res)
            'sum_{a, b, c, d} y[c]*y[d]*o[a, c]*o[b, d] * v[a] * v[b]'

        We can also make substitution on the vectors,

        .. doctest::

            >>> w = Vec('w')
            >>> res = tensor.subst(v[a], dr.einst(o[a, b] * w[b]))
            >>> str(res)
            'sum_{a, b, c, d} x[a]*x[b]*o[a, c]*o[b, d] * w[c] * w[d]'

        After the substitution, we can always make a simplification, at least to
        make the naming of the dummies more aesthetically pleasing,

        .. doctest::

            >>> res = res.simplify()
            >>> str(res)
            'sum_{a, b, c, d} x[c]*x[d]*o[c, a]*o[d, b] * w[a] * w[b]'

        """

        # Special case of the unity LHS.
        if lhs == 1:
            scalar_part = self.filter(lambda x: len(x.vecs) == 0)
            other_part = self.filter(lambda x: len(x.vecs) > 0)
            res = other_part + scalar_part * rhs
            return res

        # Shallow parsing of the left-hand side.

        if isinstance(lhs, Indexed):
            if_scalar = True
            if_indexed = True
            bases = (lhs.base,)
        elif isinstance(lhs, Symbol):
            if_scalar = True
            if_indexed = False
            bases = (lhs,)
        elif isinstance(lhs, ATerms):
            if_scalar = False
            if_indexed = True

            if len(lhs.terms) != 1:
                raise ValueError('Invalid LHS to substitute', lhs.terms)
            term = lhs.terms[0]
            if len(term.sums) != 0 or term.amp != 1:
                raise ValueError('Invalid LHS to substitute', term)
            vecs = term.vecs
            bases = {i.base for i in vecs}

            # In this way, lhs is either an SymPy expression (Indexed or Symbol)
            # or a tuple.
            lhs = vecs
        else:
            raise TypeError(
                'Invalid LHS for substitution', lhs,
                'expecting vector, indexed, or symbol'
            )

        if not all(self.has_base(i) for i in bases):
            return self

        # Shallow processing of the right-hand side.

        # We need to gather, and later broadcast all the terms.  The rational is
        # that the RHS is usually small in real problems.
        if isinstance(rhs, Tensor):
            rhs_terms = rhs.local_terms
        else:
            rhs_terms = parse_terms(rhs)

        if if_scalar and not all(
                i.is_scalar for i in rhs_terms
        ):
            raise ValueError('Invalid RHS for substituting a scalar', rhs)

        # Handling of the wilds.

        if wilds is None:
            wilds = {}
            if if_indexed:
                if if_scalar:
                    index_bunches = [lhs.indices]
                else:
                    index_bunches = [i.indices for i in lhs]
                for indices in index_bunches:
                    wilds.update(
                        (i, Wild(i.name))
                        for i in indices if isinstance(i, Symbol)
                    )

        if if_scalar:
            lhs = lhs.xreplace(wilds)
        else:
            lhs = tuple(
                i.map(lambda x: x.xreplace(wilds)) for i in lhs
            )

        rhs_terms = [j.subst(wilds) for i in rhs_terms for j in i.expand()]

        # Processing of the expression to be substituted.
        #
        # Here we manipulate lhs and create target and restore.

        decr_suffix = 'InternalProxy'

        if not simult:

            target = self
            restore = None

        elif if_scalar and not if_indexed:

            assert isinstance(lhs, Symbol)
            orig_lhs = lhs
            decored_lhs = Symbol(lhs.name + decr_suffix)
            target = self.map2amps(lambda x: x.xreplace({
                orig_lhs: decored_lhs
            }))
            lhs = decored_lhs

            # Not possible for the target to be failed to be matched against.
            restore = None

        elif if_scalar:

            assert isinstance(lhs, Indexed)
            orig_base = lhs.base
            decored_base = IndexedBase(orig_base.label.name + decr_suffix)

            def decr(x: Expr):
                return x.xreplace({
                    orig_base: decored_base
                })

            target = self.map2amps(decr)
            lhs = decr(lhs)

            def restore(t: Term):
                return t.map(lambda x: x.xreplace({
                    decored_base: orig_base
                }), skip_vecs=True)

        else:

            # For vectors, we decorate them by attaching a special placeholder
            # to their original labels, since vector labels has been documented
            # to be allowed to be anything.

            def decr_vecs(vs: typing.Tuple[Vec]):
                return tuple(
                    Vec(label=(i.label, _Decred(0)), indices=i.indices)
                    for i in vs
                )

            target = self.map(lambda t: t.map(vecs=decr_vecs(t.vecs)))
            assert isinstance(lhs, tuple)
            lhs = decr_vecs(lhs)

            def restore(t: Term):
                # Only restore the decorated vectors.
                return t.map(vecs=tuple(
                    Vec(label=i.label[0], indices=i.indices)
                    if isinstance(i.label, tuple) and len(i.label) == 2
                       and isinstance(i.label[1], _Decred)
                    else i
                    for i in t.vecs
                ))

        target = target.expand()

        # Invoke the core facility.
        res = target._subst(
            lhs, rhs_terms, full_balance=full_balance, excl=excl
        )

        # Restore
        if restore is None:
            return res
        else:
            return res.map(restore)

    def _subst(
            self, lhs: typing.Union[typing.Tuple[Vec], Indexed, Symbol],
            rhs_terms, full_balance, excl=None
    ):
        """Core substitution function.

        This function assumes the self and the substituting terms are already
        fully expanded.  And the LHS and the RHS of the substitution have been
        replaced with the wilds if it is needed.
        """

        free_vars_local = (
                self.free_vars | set.union(*[i.free_vars for i in rhs_terms])
        )
        if excl is not None:
            free_vars_local |= excl

        free_vars = self._drudge.ctx.broadcast(free_vars_local)
        dumms = self._drudge.dumms
        full_simplify = self._drudge.full_simplify

        # We keep the dummbegs dictionary for each term and substitute all
        # appearances of the lhs one-by-one.

        subs_states = self._terms.map(lambda x: x.reset_dumms(
            dumms=dumms.value, excl=free_vars.value
        ))

        rhs_terms = self._drudge.ctx.broadcast(rhs_terms)

        if isinstance(lhs, (Indexed, Symbol)):
            res = nest_bind(subs_states, lambda x: subst_factor_term(
                x[0], lhs, rhs_terms.value,
                dumms=dumms.value, dummbegs=x[1], excl=free_vars.value,
                full_simplify=full_simplify
            ), full_balance=full_balance)
        else:
            res = nest_bind(subs_states, lambda x: subst_vec_term(
                x[0], lhs, rhs_terms.value,
                dumms=dumms.value, dummbegs=x[1], excl=free_vars.value
            ), full_balance=full_balance)

        res_terms = res.map(operator.itemgetter(0))
        return Tensor(
            self._drudge, res_terms, free_vars=free_vars_local, expanded=True
        )

    def subst_all(
            self, defs, simplify=False, full_balance=False, excl=None,
            simult=True
    ):
        """Substitute all given definitions serially.

        The definitions should be given as an iterable of either
        :py:class:`TensorDef` instances or pairs of left-hand side and
        right-hand side of the substitutions.  Note that the substitutions are
        going to be performed **according to the given order** one-by-one,
        rather than simultaneously.
        """

        res = self
        for i in defs:
            if isinstance(i, TensorDef):
                lhs = i.lhs
                rhs = i.rhs
            elif isinstance(i, Sequence) and len(i) == 2:
                lhs, rhs = i
            else:
                raise TypeError(
                    'Invalid substitution', i,
                    'expecting definition or LHS/RHS pair'
                )

            res = res.subst(
                lhs, rhs, full_balance=full_balance, excl=excl, simult=simult
            )
            if simplify:
                res = res.simplify().repartition()

            # Mostly to make the evaluation eagerly.
            if res.n_terms == 0:
                return res

        return res

    def rewrite(self, vecs, new_amp):
        """Rewrite terms with the given vectors in terms of the new amplitude.

        This method will rewrite the terms whose vector part patches the given
        vectors in terms of the given new amplitude.  And all terms rewritten
        into the same form will be aggregated into a single term.

        Parameters
        ----------

        vecs
            A vector or a product of vectors.  They should be written in terms
            of SymPy wild symbols when they need to be matched against different
            actual vectors.

        new_amp
            The amplitude that the matched terms should have.  They are usually
            written in terms of the same wild symbols as the wilds in the
            vectors.

        Returns
        -------

        rewritten
            The tensor with the requested terms rewritten in term of the given
            amplitude.

        defs
            The actual definitions of the rewritten amplitude.  One for each
            rewritten term in the result.

        """

        vecs_terms = parse_terms(vecs)
        invalid_vecs = ValueError(
            'Invalid vectors to rewrite', vecs,
            'expecting just vectors'
        )
        if len(vecs_terms) != 1:
            raise invalid_vecs
        vecs_term = vecs_terms[0]
        if len(vecs_term.sums) > 0 or vecs_term.amp != 1:
            raise invalid_vecs
        vecs = vecs_term.vecs

        rewritten = self._terms.map(
            lambda term: rewrite_term(term, vecs, new_amp)
        ).cache()
        new_terms = [
            i for i in rewritten.countByKey().keys() if i is not None
        ]

        get_term = operator.itemgetter(1)
        untouched_terms = rewritten.filter(
            lambda x: x[0] is None
        ).map(get_term)
        new_defs = {}
        for i in new_terms:
            def_terms = rewritten.filter(lambda x: x[0] == i).map(get_term)
            # Eagerly evaluate to circumvent a Spark bug.
            def_terms.cache()
            def_terms.count()
            new_defs[i.amp] = Tensor(self._drudge, def_terms)
            continue

        return Tensor(self._drudge, untouched_terms.union(
            self._drudge.ctx.parallelize(new_terms)
        )), new_defs

    def expand_sums(self, range_: Range, expander: Sum_expander):
        """Expand some symbolic summations.

        The basic structure of drudge always treat summation dummies as
        primitive scalar quantities.  However, sometimes, we would want to use a
        single summation over an abstract domain to symbolically denote multiple
        simple summations.

        Rather than adding this to the base system with possibly significant
        complication of the code base, here we seek a pragmatic solution, where
        we can just use plain symbols to denote complex summations, and this
        function can be used to expand the summation into its primitive
        components when we need to.

        Parameters
        ----------

        range_
            The range all summations over which are to be expanded.

        expander
            A callable to give information how the summation over a dummy should
            be expanded.  It will be called with a dummy that is summed over the
            given range, and it is expected to return an iterable of triples,
            formed by the new dummy and new range, followed by previous form the
            the dummy in the expression, which will be replaced by the new
            dummy.  Note that after the substitution, all the original dummies
            over the expanded range should no longer be present in the
            expression, or an error will be raised.

        """
        return self.map(functools.partial(
            expand_sums_term, range_=range_, expander=expander
        ))

    #
    # Analytic gradient
    #

    def diff(self, variable, real=False, wirtinger_conj=False):
        """Differentiate the tensor to get the analytic gradient.

        By this function, support is provided for evaluating the derivative with
        respect to either a plain symbol or a tensor component.  This is
        achieved by leveraging the core differentiation operation to SymPy.  So
        very wide range of expressions are supported.

        .. warning::

            For non-analytic complex functions, this function gives the
            Wittinger derivative with respect to the given variable only.  The
            other non-vanishing derivative with respect to the conjugate needs
            to be evaluated by another invocation with ``wittinger_conj`` set to
            true.

        .. warning::

            The differentiation algorithm currently does **not** take the
            symmetry of the tensor to be differentiated with respect to into
            account.  For differentiate with respect to symmetric tensor,
            further symmetrization of the result might be needed.


        Parameters
        ----------

        variable
            The variable to differentiate with respect to.  It should be either
            a plain SymPy symbol or a indexed quantity.  When it is an indexed
            quantity, the indices should be plain symbols with resolvable range.

        real : bool
            If the variable is going to be assumed to be real.  Real variables
            has conjugate equal to themselves.

        wirtinger_conj : bool
            If we evaluate the Wirtinger derivative with respect to the
            conjugate of the variable.

        """

        if real and wirtinger_conj:
            raise ValueError(
                'Wittinger conjugate derivative vanishes for real variables'
            )

        if isinstance(variable, Indexed):

            symms = self._drudge.symms.value
            if_symm = (
                    variable.base in symms or
                    (variable.base, len(variable.indices)) in symms
            )
            if if_symm:
                warnings.warn(
                    'Gradient wrt to symmetric tensor {} '.format(variable) +
                    'might need further symmetrization'
                )

            # We need a copy.
            excl = set(self.free_vars)

            for i in variable.indices:
                if not isinstance(i, Symbol):
                    raise ValueError(
                        'Invalid index', i, 'expecting plain symbol'
                    )
                if i in excl:
                    raise ValueError(
                        'Invalid index', i,
                        'clashing with existing free symbols'
                    )
                excl.add(i)
                continue

            terms = self._reset_dumms(self._terms, excl=excl)

        elif isinstance(variable, Symbol):
            terms = self._terms
        else:
            raise TypeError('Invalid variable to differentiate', variable)

        return Tensor(self._drudge, self._diff(
            terms, variable, real=real, wirtinger_conj=wirtinger_conj
        ), expanded=True)

    def _diff(self, terms, variable, real, wirtinger_conj):
        """Differentiate the terms."""

        diff = terms.map(
            lambda x: diff_term(x, variable, real, wirtinger_conj)
        )
        # We have got to simplify here to avoid confuse users.
        return self._simplify_deltas(diff, False)

    #
    # Advanced manipulations.
    #

    def filter(self, crit):
        """Filter out terms satisfying the given criterion.

        The criterion needs to be a callable accepting :py:class:`Term`
        objects.  In the result, only terms for which the given criterion
        function evaluates to True will be retained.
        """
        return self.apply(
            lambda terms: terms.filter(crit),
            free_vars=None, repartitioned=False
        )

    def map(self, func):
        """Map the given function to the terms in the tensor.

        The given function should take a :py:class:`Term` object and return a
        :py:class:`Term` object.  The resulted tensor has the result of the
        application of the given callable to each of the terms as its terms.
        """
        return Tensor(self._drudge, self._terms.map(func))

    def bind(self, func):
        """Map the given function to the terms and flatten.

        The given callable need to return an iterable of :py:class:`Term`
        objects for each of the terms in the current tensor.  It is going to
        be applied to each of the terms, all the terms from all the results
        is going to form the terms of the resulted tensor.  This is the
        ``bind`` operation for the monad of lists in Haskell, or ``flatMap``
        in Java/Scala/Spark collection API.

        :py:meth:`filter` and :py:meth:`map` can be understood as special
        case of this method.
        """
        return Tensor(self._drudge, self._terms.flatMap(func))

    def map2scalars(self, action, skip_vecs=False, skip_ranges=True):
        """Map the given action to the scalars in the tensor.

        The given action should return SymPy expressions for SymPy expressions,
        the amplitude for each terms and the indices to the vectors, in the
        tensor.  Note that this function is more supposed for free variables and
        does not change the summation dummies in the terms.

        Parameters
        ----------

        action
            The callable to be applied to the scalars inside the tensor.

        skip_vecs
            When it is set, the callable will no longer be mapped to the indices
            to the vectors.  It could be used to boost the performance when we
            know that the action need no application on the indices.

        skip_ranges
            When it is set, the callable will no longer be mapped to the bounds
            of the ranges.

        """

        return Tensor(self._drudge, self._terms.map(lambda x: x.map(
            action, skip_vecs=skip_vecs, skip_ranges=skip_ranges
        )))

    def map2amps(self, action):
        """Map the given action to the amplitudes in the tensor.

        This method is a plain wrapper for calling :py:meth:`map2scalars` with
        vectors skipped.
        """

        return self.map2scalars(action, skip_vecs=True)

    #
    # Operations from the drudge
    #

    def __getattr__(self, item):
        """Try to see if the item is a tensor method from the drudge.

        This enables individual drudges to dynamically add domain-specific
        operations on tensors.
        """

        try:
            meth = self._drudge.get_tensor_method(item)
        except KeyError:
            raise AttributeError('Invalid operation name on tensor', item)

        return functools.partial(meth, self)


class TensorDef(Tensor):
    """Definition of a tensor.

    A tensor definition is basically a tensor with a name.  In additional to
    being a tensor, a tensor definition also has a left-hand side.  When the
    tensor is zero-order, the left-hand side is simply a symbol.  When it has
    external indices, the base and external indices for the it are both
    stored.  The base is a :py:class:`Vec` instance for tensors with vector
    part or it is an SymPy IndexedBase for scalar tensors.  For instance,

    .. math::

        \sum_j o_{i, j} f_j

    can be construed as a tensor.  By storing it as an :py:class:`Tensor`
    object, we can have mathematical manipulations on it.  With an explicit
    left-hand side,

    .. math::

        t_j = \sum_j o_{i, j} f_j

    it is now an tensor definition, which can be handled by the current class.

    A tensor definition is a subclass of tensor.  With explicit storage of a
    left-hand side, it can be convenient to be used for :py:meth:`Tensor.subst`
    or :py:meth:`Tensor.subst_all` method, or it can be directly indexed.  For
    example, with the above definition stored in ``t_def``, for tensor ``v``
    holding :math:`\sum_j u_{i, j} t_j`,

    ::

        v.subst(t_def.lhs, t_def.rhs)

    or ::

        v.subst_all([t_def])

    or, with the :py:meth:`act` method, ::

        t_def.act(v)

    we can get

    .. math::

        \sum_{j, k} u_{i, j} o_{j, k} f_k

    Tensor definition can also be directly subscripts, like ``t_def[1]``
    gives :math:`sum_i o_{1, i} f_i`.

    By being :py:class:`Tensor` subclass, all tensor manipulations are
    supported.  Just the result will not be an automatically a definition.

    """

    __slots__ = [
        '_base',
        '_exts',
    ]

    def __init__(self, base, exts, tensor: Tensor):
        """Initialize the tensor definition.

        In the same way as the initializer for the :py:class:`Tensor` class,
        this initializer is also unlikely to be used directly in user code.
        Drudge methods :py:meth:`Drudge.define` and
        :py:meth:`Drudge.define_einst`, and their wrapper
        :py:meth:`Drudge.def_` can be more convenient.  Inside drudge
        scripts, operators ``<<=`` or ``<=`` can be used, see:ref:`drs intro`
        and :py:meth:`Drudge.exec_drs`.

        Parameters
        ----------

        base
            The base for the definition.  It will be normalized to the correct
            type depending on the external indices and the right-hand side.

        exts
            The iterable for external indices.  They can be either symbol/range
            pairs for external indices with explicit range, or they can also be
            a plain symbol for generic definitions.

        tensor
            The RHS of the definition.

        """

        if isinstance(tensor, Tensor):
            super().__init__(tensor.drudge, tensor.terms)
        else:
            raise TypeError(
                'Invalid LHS for tensor definition', tensor,
                'expecting a tensor instance'
            )

        self._exts = []
        for i in exts:
            explicit_ext = (
                    isinstance(i, Sequence) and len(i) == 2 and
                    isinstance(i[0], Symbol) and isinstance(i[1], Range)
            )
            if explicit_ext:
                self._exts.append(tuple(i))
            elif isinstance(i, Expr):
                self._exts.append((
                    i, None
                ))
            else:
                raise TypeError(
                    'Invalid external index', i,
                    'expecting dummy/range pair or a dummy.'
                )
            continue

        # Normalize the base.
        base_name = str(base)
        is_scalar = self.is_scalar
        if base == 1:
            # Special case for identity definition.
            self._base = base
        elif is_scalar and len(self._exts) == 0:
            self._base = Symbol(base_name)
        elif is_scalar:
            self._base = IndexedBase(base_name)
        else:
            self._base = Vec(base_name)

    #
    # Basic properties.
    #

    @property
    def rhs(self):
        """Get the right-hand-side of the definition.

        The result is the definition itself.  Kept here for backward
        compatibility.
        """
        return self

    @property
    def rhs_terms(self):
        """Gather the terms on the right-hand-side of the definition."""
        return self.local_terms

    @property
    def lhs(self):
        """Get the standard left-hand-side of the definition."""
        if len(self._exts) == 0:
            return self._base
        else:
            return self._base[tuple(i[0] for i in self._exts)]

    @property
    def base(self):
        """The base of the tensor definition."""
        return self._base

    @property
    def exts(self):
        """The external indices."""
        return self._exts

    #
    # Simple operations.
    #

    def simplify(self):
        """Simplify the tensor in the definition.
        """

        reset = self.reset_dumms()
        return TensorDef(reset.base, reset.exts, Tensor.simplify(reset))

    def reset_dumms(self, excl=None):
        """Reset the dummies in the definition.

        The external indices will take higher precedence over the summed indices
        inside the right-hand side.
        """

        dumms = self.drudge.dumms

        exts, ext_substs, dummbegs = Term.reset_sums(
            self._exts, dumms.value, excl=excl
        )

        free_vars = self.free_vars
        if excl is None:
            excl = free_vars
        else:
            excl |= free_vars
        excl -= {i for i, _ in self._exts}

        tensor = Tensor(
            self.drudge,
            self.terms.map(lambda x: x.reset_dumms(
                dumms=dumms.value, excl=excl,
                dummbegs=dict(dummbegs), add_substs=ext_substs
            )[0])
        )

        return TensorDef(self._base, exts, tensor)

    def __eq__(self, other):
        """Compare two tensor definitions for equality.

        Note that similar to the equality comparison of tensors, here we only
        compare the syntactic equality rather than the mathematical equality.
        The left-hand side is put into consideration only for comparison with
        another tensor definition.
        """

        rhs_eq = super().__eq__(other)

        return rhs_eq and (
            self.lhs == other.lhs if isinstance(other, TensorDef) else True
        )

    #
    # Representations.
    #

    def __str__(self):
        """Form simple readable string for a definition.
        """

        return ' = '.join([str(self.lhs), super().__str__()])

    def latex(self, **kwargs):
        r"""Get the latex form for the tensor definition.

        The result will just be the form from :py:meth:`Tensor.latex` with the
        RHS prepended.

        Parameters
        ----------

        kwargs
            All keyword parameters are forwarded to the
            :py:meth:`Drudge.format_latex` method.

        """
        return self.drudge.format_latex(self, **kwargs)

    def display(self, if_return=True, **kwargs):
        """Display the tensor definition in interactive notebook sessions.

        The parameters here all have the same meaning as in
        :py:meth:`Tensor.display`.
        """

        form = Math(self.latex(**kwargs))
        if if_return:
            return form
        else:
            display(form)
            return

    #
    # Substitution.
    #

    def act(self, tensor, wilds=None, full_balance=False):
        """Act the definition on a tensor.

        This method is the active voice version of the :py:meth:`Tensor.subst`
        function.  All appearances of the defined object in the tensor will be
        substituted.

        """

        if not isinstance(tensor, Tensor):
            tensor = self.rhs.drudge.sum(tensor)

        return tensor.subst(
            self.lhs, self.rhs, wilds=wilds, full_balance=full_balance
        )

    def __getitem__(self, item):
        """Get the tensor when the definition is indexed.
        """

        if not isinstance(item, Sequence):
            item = (item,)

        n_exts = len(self._exts)
        if len(item) != n_exts:
            raise ValueError(
                'Invalid subscripts', item, 'expecting', n_exts
            )

        return self.act(self._base[item])

    #
    # Pickling support
    #

    def __getstate__(self):
        """Get the current state of the definition.
        """
        return self._base, self._exts, self.local_terms

    def __setstate__(self, state):
        """Set the state for the new definition.
        """
        super().__setstate__(state[2])
        self.__init__(state[0], state[1], self)
        return


class Drudge:
    """The main drudge class.

    A drudge is a robot who can help you with the menial tasks of symbolic
    manipulation for tensorial and noncommutative alegbras.  Due to the
    diversity and non-uniformity of tensor and noncommutative algebraic
    problems, to set up a drudge, domain-specific information about the problem
    needs to be given.  Here this is a base class, where the basic operations
    are defined. Different problems could subclass this base class with
    customized behaviour.  Most importantly, the method :py:meth:`normal_order`
    should be overridden to give the commutation rules for the algebraic system
    studied.

    """

    # We do not need slots here.  There is generally only one drudge instance.

    def __init__(self, ctx: SparkContext, num_partitions=True):
        """Initialize the drudge.

        Parameters
        ----------

        ctx
            The Spark context to be used.

        num_partitions
            The preferred number of partitions.  By default, it is the default
            parallelism of the given Spark environment.  Or an explicit integral
            value can be given.  It can be set to None, which disable all
            explicit load-balancing by shuffling.

        """

        self._ctx = ctx

        if num_partitions is True:
            self._num_partitions = self._ctx.defaultParallelism
        elif isinstance(num_partitions, int) or num_partitions is None:
            self._num_partitions = num_partitions
        else:
            raise TypeError('Invalid default partition', num_partitions)

        self._full_simplify = True
        self._simple_merge = False

        self._default_einst = False

        self._dumms = BCastVar(self._ctx, {})
        self._symms = BCastVar(self._ctx, {})
        self._resolvers = BCastVar(self._ctx, [])

        self._names = types.SimpleNamespace()

        self._tensor_methods = {}

        self._drs_specials = types.SimpleNamespace()
        self._drs_specials.S = DrsSymbol
        self._drs_specials.sum_ = sum
        self._drs_specials.DRUDGE = self

        self._inside_drs = False

    @property
    def ctx(self):
        """The Spark context of the drudge.
        """
        return self._ctx

    #
    # General settings
    #

    @property
    def num_partitions(self):
        """The preferred number of partitions for data.
        """
        return self._num_partitions

    @num_partitions.setter
    def num_partitions(self, value):
        """Set the preferred number of partitions for data.
        """
        if isinstance(value, int) or value is None:
            self._num_partitions = value
        else:
            raise TypeError(
                'Invalid default number of partitions', value,
                'expecting integer or None'
            )

    @property
    def full_simplify(self):
        """If full simplification is to be performed on amplitudes.

        It can be used to disable full simplification of the amplitude
        expression by SymPy.  For simple polynomial amplitude, this option is
        generally safe to be disabled.
        """
        return self._full_simplify

    @full_simplify.setter
    def full_simplify(self, value):
        """Set if full simplification is going to be carried out.
        """
        if value is not True and value is not False:
            raise TypeError(
                'Invalid full simplification option', value,
                'expecting boolean'
            )
        self._full_simplify = value

    @property
    def simple_merge(self):
        """If only simple merge is to be carried out.

        When it is set to true, only terms with same factors involving dummies
        are going to be merged.  This might be helpful for cases where the
        amplitude are all simple polynomials of tensorial quantities.  Note that
        this could disable some SymPy simplification.

        .. warning::

            This option might not give much more than disabling full
            simplification but taketh away many simplifications.  It is in
            general not recommended to be used.

        """
        return self._simple_merge

    @simple_merge.setter
    def simple_merge(self, value):
        """Set if simple merge is going to be carried out.
        """

        if value is not True and value is not False:
            raise ValueError(
                'Invalid simple merge setting', value,
                'expecting plain boolean'
            )
        self._simple_merge = value

    @property
    def default_einst(self):
        """If :py:meth:`def_` takes Einstein convention.

        This property tunes the behaviour of :py:meth:`def_`.  When it is
        set, the Einstein summation convention is always assumed for the
        right-hand side for that function.
        """
        return self._default_einst

    @default_einst.setter
    def default_einst(self, value):
        """Set if Einstein convention definition is default for def_.
        """

        if value is not True and value is not False:
            raise ValueError(
                'Invalid default Einstein convention', value,
                'expecting plain boolean'
            )
        self._default_einst = value

    #
    # Name archive utilities.
    #

    def form_base_name(self, tensor_def: TensorDef) -> typing.Optional[str]:
        """Form the name for the base to use for tensor definitions.

        This method is called by :py:meth:`set_name` to get a formatted string
        for the base of the tensor definition, which is to be used as the
        name for the base in the name archive.  ``None`` can be returned to
        stop the base from being added.

        By default, an underscore is put in front of the string form of the
        base.

        """
        if self is not tensor_def.drudge:
            raise ValueError('Unable to add definition from another drudge.')

        return '_' + str(tensor_def.base)

    def form_def_name(self, tensor_def: TensorDef) -> typing.Optional[str]:
        """Form the name for a tensor definition in name archive.

        The result will be used by :py:meth:`set_name` as the name of the
        tensor definition itself in the name archive.  By default, it is set
        just to be plain string form of the base of the definition.

        """
        if self is not tensor_def.drudge:
            raise ValueError('Unable to add definition from another drudge.')
        return str(tensor_def.base)

    def set_name(self, *args, **kwargs):
        """Set objects into the name archive of the drudge.

        For positional arguments, the str form of the given label is going to
        be used for the name of the object.  Special treatment is given to
        tensor definitions, the base and and definition itself will be added
        under names given by the methods :py:meth:`form_base_name`,
        and :py:meth:`form_def_name`.

        For keyword arguments, the keyword will be used for the name.
        """

        for label, obj in self._get_name_obj_pairs(args, kwargs):
            setattr(self._names, label, obj)
            continue
        return

    def unset_name(self, *args, **kwargs):
        """Unset names from name archive.

        This method is mostly used to undo the effect of :py:meth:`set_name`.
        Here, names that are not actually present in the name archive will be
        skipped without error.

        """

        for label, obj in self._get_name_obj_pairs(args, kwargs):
            if hasattr(self.names, label):
                delattr(self._names, label)
            continue
        return

    def _get_name_obj_pairs(self, args, kwargs):
        """Get the name/object pairs for name archive.

        The pairs are generated lazily.
        """

        for i in args:
            if isinstance(i, TensorDef):
                for j, k in [
                    (self.form_base_name(i), i.base),
                    (self.form_def_name(i), i)
                ]:
                    if j is not None:
                        yield j, k
            else:
                yield str(i), i
            continue

        yield from kwargs.items()

    @property
    def names(self):
        """The name archive for the drudge.

        The name archive object can be used for convenient accessing of objects
        related to the problem.

        """
        return self._names

    def inject_names(self, prefix='', suffix=''):
        """Inject the names in the name archive into the current global scope.

        This function is for the convenience of users, especially interactive
        users.  Itself is not used in official drudge code except its own tests.

        Note that this function injects the names in the name archive into the
        **global** scope of the caller, rather than the local scope, even when
        called inside a function.
        """

        # Find the global scope for the caller.
        stack = inspect.stack()
        try:
            globals_ = stack[1][0].f_globals
        finally:
            del stack

        for k, v in self._names.__dict__.items():
            globals_[''.join([prefix, k, suffix])] = v

        return

    #
    # General properties
    #
    # Subclasses normally just need to use the methods in this section to add
    # some additional information.  The method here generally does not need to
    # be overridden.
    #

    def set_dumms(self, range_: Range, dumms,
                  set_range_name=True, dumms_suffix='_dumms',
                  set_dumm_names=True):
        """Set the dummies for a range.

        Note that this function overwrites the existing dummies if the range has
        already been given.
        """

        new_dumms = [ensure_symb(i) for i in dumms]
        self._dumms.var[range_] = new_dumms

        if set_range_name:
            self.set_name(range_)
        if dumms_suffix:
            self.set_name(**{str(range_) + dumms_suffix: new_dumms})
        if set_dumm_names:
            for i in new_dumms:
                self.set_name(i)

        return new_dumms

    @property
    def dumms(self):
        """The broadcast form of the dummies dictionary.
        """
        return self._dumms.bcast

    def set_symm(self, base, *symms, valence=None, set_base_name=True):
        """Set the symmetry for a given base.

        Permutation objects in the arguments are interpreted as single
        generators, other values will be attempted to be iterated over to get
        their entries, which should all be permutations.

        Parameters
        ----------

        base
            The SymPy indexed base object or vectors whose symmetry is to be
            set.  Their label can be used as well.

        symms
            The generators of the symmetry.  It can be a single None to remove
            the symmetry of the given base.

        valence : int
            When it is set, only the indexed quantity of the base with the given
            valence will have the given symmetry.

        set_base_name
            If the base name is to be added to the name archive of the drudge.

        """

        if len(symms) == 0:
            raise ValueError('Invalid empty symmetry, expecting generators!')
        elif len(symms) == 1 and symms[0] is None:
            group = None
        else:
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

            if len(gens) > 0:
                group = Group(gens)
            else:
                group = None

        valid_valence = (valence is None or (
                isinstance(valence, int) and valence > 1
        ))
        if not valid_valence:
            raise ValueError(
                'Invalid valence', valence, 'expecting positive integer'
            )

        self._symms.var[
            base if valence is None else (base, valence)
        ] = group

        if set_base_name:
            self.set_name(**{str(base): base})

        return group

    @property
    def symms(self):
        """The broadcast form of the symmetries.
        """
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

    def add_resolver_for_dumms(self, ranges=None, strict=False):
        """Add the resolver for the dummies.

        With this method, the default dummies for each range will be resolved to
        be within the range for all of them.  This method should normally be
        called by all subclasses after the dummies for all ranges have been
        properly set.

        Note that dummies added later will not be automatically added.  This
        method can be called again.

        Parameters
        ----------

        ranges
            By default, resolver for all ranges will be attempted to be added.
            Or a specific iterable of ranges can also be given to add resolver
            only for the given ranges.

        strict
            By default, any expression contains a dummy of the range will be
            considered to be of the range.  If strict is set to true, only the
            dummy appearing by themself will be resolved.


        """

        # Normalize the ranges to iterable of range/dummies pairs.
        curr_dumms = self._dumms.ro.items()
        if ranges is None:
            to_proc = curr_dumms
        else:
            to_proc = []
            for i in ranges:
                if i not in curr_dumms:
                    raise ValueError(
                        'Unexpected range, dummies not yet set', i
                    )
                to_proc.append((i, curr_dumms[i]))
                continue

        self.add_resolver(SymbResolver(to_proc, strict))

    def add_default_resolver(self, range_):
        """Add a default resolver.

        The default resolver will resolve *any* expression to the given range.
        Note that all later resolvers will not be invoked at all after this
        resolver is added.
        """
        self.add_resolver(functools.partial(
            _resolve_default_range, range_=range_
        ))

    @property
    def resolvers(self):
        """The broadcast form of the resolvers."""
        return self._resolvers.bcast

    def set_tensor_method(self, name, func):
        """Set a new tensor method under the given name.

        A tensor method is a method that can be called from tensors created from
        the current drudge as if it is a method of the given tensor. This could
        give cleaner and more consistent code for all tensor manipulations.

        The given function, or bounded method, should be able to accept the
        tensor as the first argument.
        """

        self._tensor_methods[name] = func

    def get_tensor_method(self, name):
        """Get a tensor method with given name.

        When the name cannot be resolved, KeyError will be raised.
        """

        return self._tensor_methods[name]

    #
    # Vector-related properties
    #
    # The methods here is highly likely to be overridden for different
    # non-commutative algebraic systems in different problems.
    #

    @property
    def vec_colour(self):
        """The vector colour function.

        Note that this accessor accesses the **function**, rather than directly
        computes the colour for any vector.
        """
        return None

    def normal_order(self, terms, **kwargs):
        """Normal order the terms in the given tensor.

        This method should be called with the RDD of some terms, and another RDD
        of terms, where all the vector parts are normal ordered according to
        domain-specific rules, should be returned.

        By default, we work for the free algebra.  So nothing is done by this
        function.  For noncommutative algebraic system, this function needs to
        be overridden to return an RDD for the normal-ordered terms from the
        given terms.
        """

        if len(kwargs) != 0:
            raise ValueError(
                'Invalid arguments to free algebra normal order', kwargs
            )

        return terms

    #
    # Other tunable behaviour
    #

    @staticmethod
    def simplify_amp_sum(expr: Sum):
        """The default callable to simplify summations within amplitudes.

        This is going to be retrieved as an attribute of the drudge object, and
        the result should be a callable (serializable for parallel execution)
        object capable of attempting to simplify summations within amplitudes,
        which will be used as the default callable in
        :py:meth:`Tensor.simplify_sums`.

        By default, only the SymPy ``eval_sum_symbolic`` function will be called
        with simplified summand when there is a single summation.  It might
        later be updated to higher-level functions when SymPy bug #13979 is
        resolved.
        """
        assert isinstance(expr, Sum)
        if len(expr.args) == 2:
            return eval_sum_symbolic(expr.args[0].simplify(), expr.args[1])
        else:
            return expr

    #
    # Tensor creation
    #

    def sum(self, *args, predicate=None) -> Tensor:
        """Create a tensor for the given summation.

        This is the core function for creating tensors from scratch.  The
        arguments should start with the summations, each of which should be
        given as a sequence, normally a tuple, starting with a SymPy symbol for
        the summation dummy in the first entry.  Then comes possibly multiple
        domains that the dummy is going to be summed over, which can be symbolic
        range, SymPy expression, or iterable over them.  When symbolic ranges
        are given as :py:class:`Range` objects, the given dummy will be set to
        be summed over the ranges symbolically.  When SymPy expressions are
        given, the given values will substitute all appearances of the dummy in
        the summand.  When we have multiple summations, terms in the result are
        generated from the Cartesian product of them.

        The last argument should give the actual thing to be summed, which can
        be something that can be interpreted as a collection of terms, or a
        callable that is going to return the summand when given a dictionary
        giving the action on each of the dummies.  The dictionary has an entry
        for all the dummies.  Dummies summed over symbolic ranges will have the
        actual range as its value, or the actual SymPy expression when it is
        given a concrete range.  In the returned summand, if dummies still
        exist, they are going to be treated in the same way as statically-given
        summands.

        The predicate can be a callable going to return a boolean when called
        with same dictionary.  False values can be used the skip some terms.  It
        is guaranteed that the same dictionary will be used for both predicate
        and the summand when they are given as callables.

        For instance, mostly commonly, we can create a tensor by having simple
        summations over symbolic ranges,

        .. doctest::

            >>> dr = Drudge(SparkContext())
            >>> r = Range('R')
            >>> a = Symbol('a')
            >>> b = Symbol('b')
            >>> x = IndexedBase('x')
            >>> v = Vec('v')
            >>> tensor = dr.sum((a, r), (b, r), x[a, b] * v[a] * v[b])
            >>> str(tensor)
            'sum_{a, b} x[a, b] * v[a] * v[b]'

        And we can also give multiple symbolic ranges for a single dummy to sum
        over all of them,

        .. doctest::

            >>> s = Range('S')
            >>> tensor = dr.sum((a, r, s), x[a] * v[a])
            >>> print(str(tensor))
            sum_{a} x[a] * v[a]
             + sum_{a} x[a] * v[a]

        When the objects to sum over are not symbolic ranges, we are in the
        concrete summation mode, for instance,

        .. doctest::

            >>> tensor = dr.sum((a, 1, 2), x[a] * v[a])
            >>> print(str(tensor))
            x[1] * v[1]
             + x[2] * v[2]

        The concrete and symbolic summation mode can be put together freely in
        the same summation,

        .. doctest::

            >>> tensor = dr.sum((a, r, s), (b, 1, 2), x[b, a] * v[a])
            >>> print(str(tensor))
            sum_{a} x[1, a] * v[a]
             + sum_{a} x[2, a] * v[a]
             + sum_{a} x[1, a] * v[a]
             + sum_{a} x[2, a] * v[a]

        Note that this function can also be called on existing tensor objects
        with the same semantics on the terms.  Existing summations are not
        touched by it.  For instance,

        .. doctest::

            >>> tensor = dr.sum(x[a] * v[a])
            >>> str(tensor)
            'x[a] * v[a]'
            >>> tensor = dr.sum((a, r), tensor)
            >>> str(tensor)
            'sum_{a} x[a] * v[a]'

        where we have used summation with only summand (no sums) to create
        simple tensor of only one term without any summation.

        """

        if len(args) == 0:
            raise ValueError('Expecing summand!')

        summand = args[-1]
        sum_args = args[:-1]

        if isinstance(summand, Tensor):
            return Tensor(self, summand.terms.flatMap(
                lambda x: sum_term(sum_args, x, predicate=predicate)
            ))
        else:
            return self.create_tensor(sum_term(
                sum_args, summand, predicate=predicate
            ))

    def einst(self, summand, auto_exts: bool = False) -> typing.Union[
        Tensor, typing.Tuple[Tensor, typing.AbstractSet[Symbol]]
    ]:
        """Create a tensor from Einstein summation convention.

        By calling this function, summations according to the Einstein summation
        convention will be added to the terms.  Note that this function is
        mostly designed for the most conventional tensor problems.  For a symbol
        to be recognized as a summation, it must appear twice or more in its
        **original form** in indices, and its range needs to be able to be
        resolved.  When a symbol is used in compound form as tensor indices, it
        will be considered to be for non-conventional purposes and will no
        longer be automatically recognized as summations.

        For instance, we can have the following fairly conventional Einstein
        form,

        .. doctest::

            >>> dr = Drudge(SparkContext())
            >>> r = Range('R')
            >>> a, b, c = dr.set_dumms(r, symbols('a b c'))
            >>> dr.add_resolver_for_dumms()
            >>> x = IndexedBase('x')
            >>> tensor = dr.einst(x[a, b] * x[b, c])
            >>> str(tensor)
            'sum_{b} x[a, b]*x[b, c]'

        Note that in addition to creating tensors from scratch, this method can
        also be called on an existing tensor to add new summations.  In that
        case, no existing summations will be touched.

        For indices whose range is resolved to a tuple of multiple ranges, the
        term will be expanded to sum over all the given ranges.

        Parameters
        ----------

        summand

            The summand to be applied the Einstein summation convention.

        auto_exts

            If external indices will tried to be recognized automatically and
            returned as a second return value.  For each term, symbols appearing
            exactly once **in its raw form as indices** will be considered as
            external indices.  When the set of external indices are the same for
            all terms, it will be returned as the second return value, or a
            ``ValueError`` will be raised.

        """

        resolvers = self.resolvers

        # Separate the cases for local and distributed terms for optimization.
        if isinstance(summand, Tensor):

            einst_res = summand.expand().terms.map(
                lambda x: einst_term(x, resolvers.value)
            ).cache()
            tensor = Tensor(
                self, einst_res.flatMap(operator.itemgetter(0)), expanded=True
            )

            if not auto_exts:
                exts_union = None
                exts_inters = None
            else:

                def seq_op(curr, new):
                    """Merge current externals with a new Einstein result."""
                    curr[0].update(new[1])
                    curr[1] = _inters(curr[1], new[1])
                    return curr

                def comb_op(curr, new):
                    """Merge externals from different partitions."""
                    curr[0].update(new[0])
                    curr[1] = _inters(curr[1], new[1])
                    return curr

                exts_union, exts_inters = einst_res.aggregate(
                    [set(), None], seq_op, comb_op
                )

        else:
            res_terms = []
            exts_union = set()
            exts_inters = None

            for i in parse_terms(summand):
                # We need to expand the possibly parenthesized user input.
                for j in i.expand():
                    terms, exts = einst_term(j, resolvers.value)
                    res_terms.extend(terms)
                    exts_union |= exts
                    exts_inters = _inters(exts_inters, exts)
                    continue
                continue
            tensor = self.create_tensor(res_terms)

        # End splitting between distributed and local input.

        if not auto_exts:
            return tensor

        if exts_union != exts_inters:
            diff = exts_union - exts_inters
            raise ValueError(
                'Invalid external indices for Einstein convention',
                diff, 'appeared only in some terms'
            )

        return tensor, exts_inters

    def create_tensor(self, terms):
        """Create a tensor with the terms given in the argument.

        The terms should be given as an iterable of Term objects.  This function
        should not be necessary in user code.
        """
        return Tensor(self, self._ctx.parallelize(terms))

    #
    # Tensor definition creation.
    #

    def define(self, *args) -> TensorDef:
        """Make a tensor definition.

        This is a helper method for the creation of :py:class:`TensorDef`
        instances.

        Parameters
        ----------

        initial arguments
            The left-hand side of the definition.  It can be given as an indexed
            quantity, either SymPy Indexed instances or an indexed vector, with
            all the plain symbol indices having their range is able to be
            resolved.  Or a base can be given, followed by the symbol/range
            pairs for the external indices.

        final argument
            The definition of the LHS, can be tensor instances, or anything
            capable of being interpreted as such.  Note that no summation is
            going to be automatically added.

        """
        if len(args) == 0:
            raise ValueError('Expecting arguments for definition.')

        base, exts = self._parse_def_lhs(args[:-1])
        content = args[-1]
        tensor = content if isinstance(content, Tensor) else self.sum(content)
        return TensorDef(base, exts, tensor)

    def define_einst(self, *args, auto_exts=False) -> TensorDef:
        """Make a tensor definition based on Einstein summation convention.

        This method basically performs function very similar to the
        :py:meth:`define` method, just the content will be interpreted according
        to the Einstein summation convention.  When the argument ``auto_exts``
        is set, the lhs of the definition is read in the same way as the
        :py:meth:`define` method.  When it is turned on, the lhs can be given
        only by a base and the external indices will be attempted to be
        inferred, as described in :py:meth:`einst`.

        """

        if len(args) == 0:
            raise ValueError('Expecting arguments for definition.')

        if auto_exts:
            if len(args) != 2:
                raise TypeError(
                    'Invalid number of arguments, base and rhs expected!'
                )
            tensor, exts = self.einst(args[-1], auto_exts=True)
            exts = self._form_exts(exts)
            exts.sort(key=lambda x: (x[1], sympy_key(x[0])))
            base = args[0]
        else:
            tensor = self.einst(args[-1])
            base, exts = self._parse_def_lhs(args[:-1])

        return TensorDef(base, exts, tensor)

    def _parse_def_lhs(self, args):
        """Parse the user-given LHS of tensor definitions.

        Here, very shallow checking is performed.  Detailed sanitation are to be
        performed in the tensor definition initializer.
        """

        if len(args) == 0:
            raise ValueError('No LHS given for tensor definition.')
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, (Indexed, Vec)):
                base = arg.base
                exts = self._form_exts(arg.indices)
                return base, exts
            else:
                return arg, ()
        else:
            return args[0], args[1:]

    def _form_exts(self, indices):
        """Form dummy/range pairs from symbols.

        For plain symbols, the range is resolved based on the resolver, or
        ValueError will be raised.  Currently, when it is resolved to have
        multiple ranges, it will be treated as a generic one without explicit
        range.
        """
        exts = []
        for i in indices:
            if isinstance(i, Symbol):
                range_ = try_resolve_range(i, {}, self.resolvers.value)
                if range_ is None:
                    raise ValueError(
                        'Invalid index', i, 'range cannot be resolved'
                    )
                if isinstance(range_, Sequence):
                    exts.append(i)
                else:
                    exts.append((i, range_))
            else:
                exts.append(i)
            continue
        return exts

    def def_(self, *args) -> TensorDef:
        """Make a tensor definition according to convention set in drudge.

        This method is a convenient utility for making tensor definitions.
        Basically it calls :py:meth:`define` or :py:meth:`define_einst`
        according to the value of the property :py:meth:`default_einst`.

        It is also the operations used for tensor definition operations inside
        drudge scripts.

        """

        if self.default_einst:
            return self.define_einst(*args)
        else:
            return self.define(*args)

    #
    # Printing
    #

    def format_latex(
            self, inp, sep_lines=False, align_terms=False, proc=None,
            no_sum=False, scalar_mul=''
    ):
        r"""Get the LaTeX form of a given tensor or tensor definition.

        Subclasses should fine-tune the appearance of the resulted LaTeX form by
        overriding methods ``_latex_sympy``, ``_latex_vec``, and
        ``_latex_vec_mul``.

        Parameters
        ==========

        inp

            The input tensor or tensor definition.

        sep_lines

            If terms should be put into separate lines by separating them with
            ``\\``.

        align_terms

            If ``&`` is going to be prepended to each term to have them aligned.
            This option is intended for cases where the LaTeX form is going to
            be put inside environments supporting alignment.

        proc

            A callable to be called with the string of the original LaTeX
            formatting of each of the terms to return a processed final form.
            The callable is also going to be given keyword arguments ``term``
            for the actual tensor term and ``idx`` for the index of the term
            within the tensor.

        no_sum : bool

            If summation is going to be suppressed in the printing, useful for
            cases where a convention, like the Einstein's, exists for the
            summations.

        scalar_mul : str

            The text for scalar multiplication.  By default, scalar
            multiplication is just rendered as juxtaposition.  When a string is
            given for this argument, it is going to be placed between scalar
            factors and between the amplitude and the vectors.  In LaTeX output
            of tensors with terms with many factors, special command
            ``\invismult`` can be used, which just makes a small space but
            enables the factors to be automatically split by the ``breqn``
            package.  Note that this option has effect only for expanded terms,
            where the scalar multiplication occurs on the top-level.

        """

        if isinstance(inp, TensorDef):
            prefix = (self._latex_vec(inp.lhs) if isinstance(inp.lhs, Vec) else
                      self._latex_sympy(inp.lhs)) + ' = '
        elif isinstance(inp, Tensor):
            prefix = ''
        else:
            raise TypeError('Invalid object to form into LaTeX.')

        n_terms = inp.n_terms
        inp_terms = inp.local_terms

        if n_terms == 0:
            return prefix + '0'

        terms = []
        for i, v in enumerate(inp_terms):
            term = self._latex_term(v, no_sum=no_sum, scalar_mul=scalar_mul)

            if proc is not None:
                term = proc(term, term=v, idx=i)

            if i != 0 and term[0] not in {'+', '-'}:
                term = ' + ' + term
            if align_terms:
                term = ' & ' + term
            terms.append(term)
            continue

        term_sep = r' \\ ' if sep_lines else ' '

        return prefix + term_sep.join(terms)

    def _latex_term(self, term, no_sum=False, scalar_mul=''):
        """Format a term into LaTeX form.

        This method does not generally need to be overridden.
        """

        # Small utility.
        def parenth(content: str):
            """Parenthesize the string."""
            return ''.join([r'\left(', content, r'\right)'])

        parts = []

        # Here we treat the amplitude first so that a possible minus sign can be
        # put before the term.
        amp_parts = []

        if isinstance(term.amp, Add):
            # In this case, we need a big parenthesis for the amplitude.
            first_try = self._latex_sympy(term.amp)
            if first_try[0] == '-':
                parts.append('-')
                amp_latex = self._latex_sympy(-term.amp)
            else:
                amp_latex = first_try

            amp_parts.append(parenth(amp_latex))
        else:
            factors, coeff = term.amp_factors
            if_pure_coeff = len(factors) == 0 and len(term.vecs) == 0

            if coeff == 1 and not if_pure_coeff:
                pass
            elif coeff == -1 and not if_pure_coeff:
                parts.append('-')
            else:
                coeff_latex = self._latex_sympy(coeff)
                if coeff_latex[0] == '-':
                    parts.append('-')
                    coeff_latex = coeff_latex[1:]

                if isinstance(coeff, Add):
                    coeff_latex = parenth(coeff_latex)
                amp_parts.append(coeff_latex)

            if len(factors) > 0:
                scalar_mul = ''.join([' ', scalar_mul, ' '])

                amp_parts.append(scalar_mul.join(
                    parenth(self._latex_sympy(i)) if isinstance(i, Add)
                    else self._latex_sympy(i)
                    for i in factors
                ))

        if not term.is_scalar:
            amp_parts.append(scalar_mul)

        if not no_sum:
            parts.extend(r'\sum_{{{} \in {}}}'.format(
                self._latex_sympy(i), j.label
            ) for i, j in term.sums)

        parts.append(' '.join(amp_parts))

        vecs = self._latex_vec_mul.join(
            self._latex_vec(i) for i in term.vecs
        )
        parts.append(vecs)

        return ' '.join(parts)

    @staticmethod
    def _latex_sympy(expr):
        """Get the LaTeX form of SymPy expressions.

        By default, the printing by :py:class:`ScalarLatexPrinter` will be used.
        Subclasses can override this method for fine tuning of the form.

        """
        return ScalarLatexPrinter().doprint(expr)

    def _latex_vec(self, vec):
        """Get the LaTeX form of a vector.

        By default, the vector name is going to be put into boldface, and the
        indices are put into the subscripts.
        """

        head = r'\mathbf{{{}}}'.format(vec.label)
        if len(vec.indices) > 0:
            indices = ', '.join(self._latex_sympy(i) for i in vec.indices)
            return r'{}_{{{}}}'.format(head, indices)
        else:
            return head

    _latex_vec_mul = r' \otimes '

    @contextlib.contextmanager
    def report(self, filename, title):
        """Make a report for results.

        This function should be used within a ``with`` statement to open a
        report (:py:class:`Report`) for results.

        Parameters
        ----------

        filename : str

            The name of the report file, whose extension gives the type of the
            report.  Currently, ``.html`` gives reports in HTML format, where
            the ``MathJAX`` library is used for rendering the math.  ``.tex``
            gives reports in LaTeX format, while ``.pdf`` will automatically
            compile the LaTeX source by program ``pdflatex`` in the path.
            Normally for LaTeX output, finer tuning of the display environment
            in :py:meth:`Report.add` is required, especially for long
            equations.

        title

            The title to be printed in the report.

        Examples
        --------

        .. doctest::
            :options: +SKIP

            >>> dr = Drudge(SparkContext())
            >>> tensor = dr.sum(IndexedBase('x')[Symbol('a')])
            >>> with dr.report('report.html', 'A simple tensor') as report:
            ...     report.add('Simple tensor', tensor)

        """

        report = Report(filename, title, self)
        yield report
        report.write()

    #
    # Pickling support
    #

    @contextlib.contextmanager
    def pickle_env(self):
        """Prepare the environment for unpickling contents with tensors.

        Pickled contents containing tensors cannot be directly unpickled by the
        functions from the pickle module directly.  They should be used within
        the context created by this function.  Note that the content does not
        have to have a single tensor object.  Anything containing tensor objects
        needs to be loaded within the context.

        Note that this context is unnecessary inside drudge scripts.

        .. warning::

            All tensors read within this environment will have the current
            drudge as their drudge.  No checking is, or can be, done to make
            sure that the tensors are sensible for the current drudge.  Normally
            it should be the same drudge as the drudge used for their creation
            be used.

        Examples
        --------

        .. doctest::

            >>> dr = Drudge(SparkContext())
            >>> tensor = dr.sum(IndexedBase('x')[Symbol('a')])
            >>> import pickle
            >>> serialized = pickle.dumps(tensor)
            >>> with dr.pickle_env():
            ...     res = pickle.loads(serialized)
            >>> print(tensor == res)
            True

        """

        global current_drudge
        prev_drudge = current_drudge
        current_drudge = self
        yield None
        current_drudge = prev_drudge

    def memoize(self, comput, filename, log=None, log_header='Memoize:'):
        """Preserve/lookup result of computation into/from pickle file.

        When the file with the given name exists, it will be opened and
        attempted to be unpickled, with the deserialized content returned and
        the given computation skipped.  When the file is absent or does not
        contain valid pickle, the given computation will be performed, with the
        result both pickled into a file created with the given name and
        returned.

        Parameters
        ----------

        comput

            The callable giving the computation to be performed.  To be called
            with no arguments.

        filename

            The name of the pickle file to read from or write to.

        log

            The file object to write log information to.  ``None`` if no logging
            is desired, ``True`` if they are to be written to the standard
            output, or any writable file object can be given.

        log_header

            The header to be prepended to lines of the log texts.

        Returns
        -------

        The result of the computation, either read from existing file or newly
        computed.

        Examples
        --------

        .. doctest::
            :options: +SKIP

            >>> dr = Drudge(SparkContext())
            >>> res = dr.memoize(lambda: 10, 'intermediate.pickle')
            >>> res
            10
            >>> dr.memoize(lambda: 10, 'intermediate.pickle')
            10

        Note that in the second execution, the number 10 should be read from the
        file rather than being computed again.  Normally, rather than a trivial
        number, expensive intermediate results can be memoized in this way so
        that the script can be restarted readily.

        """

        if log is True:
            log = sys.stdout
        if log is None:
            log_args = {}
        else:
            log_args = {'file': log}

        try:

            with self.pickle_env(), open(filename, 'rb') as fp:
                res = pickle.load(fp)

            print(log_header, 'read data from {}'.format(filename), **log_args)

        except (OSError, pickle.PickleError) as exc:

            print(log_header, 'computing, failed to read from {}: {!s}'.format(
                filename, exc
            ), **log_args)

            res = comput()
            with open(filename, 'wb') as fp:
                pickle.dump(res, fp)

        return res

    #
    # Drudge scripts support
    #

    @property
    def inside_drs(self):
        """If we are currently inside a drudge script."""
        return self._inside_drs

    def exec_drs(self, src, filename='<unknown>'):
        """Execute the drudge script.

        Drudge script are Python scripts tweaked to be executed in special
        environments.  This domain-specific language is made for the
        convenience users for simple tasks, especially for users unfamiliar
        with Python.

        Being a Python script executed inside the current interpreter,
        drudge script differs from normal Python scripts by

        1. All integer literal are resolved into SymPy symbolic integers.

        2. Global names are resolved in the order of,

           - the name archive in the current drudge,
           - the special drudge script functions in the drudge,
           - the drudge package exported names,
           - the gristmill package exported names (if installed),
           - the SymPy exported names,
           - built-in Python names.

        3. All unresolved names are created as a special kind of symbolic
           object, which behaves basically like SymPy ``Symbol``, but with
           differences,

           1. They are be directly subscripted, like ``IndexedBase``.

           2. ``def_as`` method can be used to make a tensor definition with
              such symbols or its indexing on the left-hand side, the other
              operand on its right-hand side.  The resulted definition is also
              added to the name archive of the drudge.

           3. ``<=`` operator can be used similar to ``def_as``, except the
              definition is not added to the archive.  The result can be put
              into local variables.

        4. All left-shift augmented assignment ``<<=`` operations are
           replaced by ``def_as`` method calling.

        5. Some operations could have slightly different behaviour more suitable
           inside drudge scripts.  For developers, the :py:meth:`inside_drs`
           property can be used to query if the function is called inside a
           drudge script.

        6. The pickling environment is set.

        For a non-technical introduction to drudge script, please see
        :ref:`drs intro`.

        """
        code = compile_drs(src, filename)
        env = DrsEnv(self, specials=self._drs_specials)
        self._inside_drs = True
        with self.pickle_env():
            exec(code, env)
        self._inside_drs = False
        return env

    def simplify(self, arg, **kwargs):
        """Make simplification for both SymPy expressions and tensors.

        This method is mostly designed to be used in drudge scripts.  But it
        can also be used inside Python scripts for convenience.

        The actual simplification is dispatched based on the type of the given
        argument. Simple SymPy simplification for SymPy expressions, drudge
        simplification for drudge tensors or tensor definitions, or the argument
        will be attempted to be converted into a tensor.

        """
        if isinstance(arg, (Tensor, Expr)):
            return arg.simplify(**kwargs)
        else:
            return self.sum(arg).simplify(**kwargs)

    #
    # Mathematical utilities.
    #

    #
    # Utilities for linear transformations of vectors.
    #

    @staticmethod
    def lvt_defs2mat(
            defs: typing.Iterable[TensorDef], rhs_vecs=None, ret_rhs=False
    ):
        r"""Turn a linear vector transformation into matrix form.

        A linear vector transformation is a set of definitions of vectors as
        simple linear combinations of another set of vectors, where the linear
        combination only has plain symbolic scalar coefficients without any
        symbolic summations.

        With this method, given a linear vector transformation as a list of
        tensor definitions, the matrix :math:`M` of the transformation will be
        formed, which makes

        .. math::

            \mathbf{L} = M \mathbf{R}

        where :math:`\mathbf{L}` is the column vector of the vectors on the
        left-hand side and :math:`\mathbf{R}` is the column vector of the
        vectors on the right-hand side.

        Parameters
        ----------

        defs
            An iterable giving all the definitions.

        rhs_vecs
            The vectors on the right-hand side.  When it is given as None, all
            vectors appearing in the RHSes of the definitions will be ordered in
            an non-deterministic way.

        ret_rhs
            If the actual list of RHS vectors is going to be returned.

        """

        # Parse the definitions to mapping from LHS to coefficients dictionary.
        coeffs = collections.OrderedDict()
        for def_ in defs:
            lhs = def_.lhs
            if lhs != 1 and not isinstance(lhs, Vec):
                raise ValueError(
                    'Invalid vector defined on LHS', lhs
                )
            if lhs in coeffs:
                raise ValueError(
                    'Duplicate definition', lhs
                )
            curr_coeffs = collections.defaultdict(lambda: 0)
            coeffs[lhs] = curr_coeffs

            for term in def_.local_terms:

                if len(term.sums) > 0:
                    raise NotImplementedError(
                        'RHS for', lhs, 'too complicated', term
                    )

                vecs = term.vecs
                if len(vecs) > 1:
                    raise ValueError(
                        'Nonlinear term for ', lhs, 'in', term
                    )
                elif len(vecs) == 1:
                    curr_coeffs[vecs[0]] += term.amp
                else:
                    curr_coeffs[1] += term.amp

                continue  # Go to next term.

            continue  # Go to the next definition.

        if rhs_vecs is None:
            rhs_vecs = set()
            for i in coeffs.values():
                rhs_vecs.update(i.keys())
                continue
            rhs_vecs = list(rhs_vecs)

        res_coeffs = []
        for lhs, rhs in coeffs.items():
            row = []
            for i in rhs_vecs:
                if i in rhs:
                    row.append(rhs.pop(i))
                else:
                    row.append(0)
                continue
            if len(rhs) != 0:
                raise ValueError(
                    'Terms with vectors not given as RHS', list(rhs.keys())
                )
            res_coeffs.append(row)
            continue

        res = Matrix(res_coeffs)
        if ret_rhs:
            return res, rhs_vecs
        else:
            return res

    def lvt_mat2defs(
            self, mat: Matrix, lhs_vecs: typing.Iterable[Vec],
            rhs_vecs: typing.Iterable[Vec]
    ):
        """Form definitions from a linear transformation matrix.

        This is the exact inverse of the operation defined in
        :py:meth:`lvt_defs2mat`.
        """

        lhs_vecs = list(lhs_vecs)
        rhs_vecs = list(rhs_vecs)
        n_rows, n_cols = mat.shape

        if n_rows != len(lhs_vecs):
            raise ValueError(
                'Expecting', n_rows, 'LHS vectors', len(lhs_vecs), 'given'
            )
        if n_cols != len(rhs_vecs):
            raise ValueError(
                'Expecting', n_cols, 'RHS vectors', len(rhs_vecs), 'given'
            )

        defs = []
        for idx in range(n_rows):
            def_ = self.define(lhs_vecs[idx], sum(
                i * j for i, j in zip(mat.row(idx), rhs_vecs)
            ))
            defs.append(def_)
            continue

        return defs

    def lvt_inv(self, defs: typing.Iterable[TensorDef], rhs_vecs=None):
        """Inverse a linear transformation of vectors.

        This method internally utilizes :py:meth:`lvt_defs2mat` to get the
        matrix for the transformation, then with the matrix inverted and
        simplified symbolically, the inverse definition is cast back to the
        definitions form.
        """

        mat, rhs_vecs = self.lvt_defs2mat(defs, rhs_vecs, ret_rhs=True)
        # Here the input should have already been checked.
        lhs_vecs = [i.lhs for i in defs]

        inv_mat = mat.inv()
        inv_mat.simplify()

        return self.lvt_mat2defs(inv_mat, rhs_vecs, lhs_vecs)


#
# Global for pickling
#


current_drudge = None


#
# Small static utilities
#


def _union(orig, new):
    """Union the two sets and return the first."""
    orig |= new
    return orig


def _inters(orig, new):
    """Make the interaction of two sets.

    If the original is a None value, a new set will be created with elements
    from the new set.  When both operands are None, the result is None.
    """
    if orig is None:
        return new
    elif new is None:
        return orig
    else:
        orig &= new
        return orig


class _DecomposeSpecials:
    """Container of special symbols during amplitude decomposition for merging.

    When the merging is in constant symbol mode, a symbol is considered to be
    contained only when it is not a constant symbol.  During generators mode, a
    symbol is considered special only when it is one of the given generators.
    In default mode, no symbol is considered special.
    """

    __slots__ = [
        '_consts',
        '_gens'
    ]

    def __init__(self, consts=None, gens=None):
        """Initialize the container."""

        self._consts = None
        self._gens = None

        if consts is not None:
            self._consts = frozenset(consts)
            assert gens is None
        elif gens is not None:
            self._gens = frozenset(gens)
            assert consts is None

    def __contains__(self, item):
        """Test if a symbol requires special treatment."""
        if self._consts is not None:
            return item not in self._consts
        elif self._gens is not None:
            return item in self._gens
        else:
            # By default, we do not give special treatment to any symbol.
            return False


def _decompose_term(term, specials):
    """Decompose a term for simple merging.

    The given term will be decomposed into a pair, where the first field has the
    the key for the merging, and the second field contains other factors.

    The key has three parts, the summations, the vectors, and the special part
    of the commutative part.  When specials are given, all indexed bases,
    factors with dummies, and factors with special symbols will be put into the
    third part.  When a None is given, nothing will be put there.
    """

    # To distinguish ranges with the same label but different bounds.
    sums = tuple(
        (dumm, range_.args) for dumm, range_ in term.sums
    )

    if specials is None:
        coeff = term.amp
        factor = 1
    else:
        factors, coeff = term.get_amp_factors(specials)
        factor = prod_(factors)
    return (
        (sums, term.vecs, factor),
        coeff
    )


def _recover_term(state):
    """Recover a term from a merging state."""

    key, coeff = state
    sums = tuple(
        (dumm, Range(*range_args)) for dumm, range_args in key[0]
    )
    return Term(sums, coeff * key[2], key[1])


def _is_nonzero(term):
    """Test if a term is trivially non-zero."""
    return term.amp != 0


def _resolve_default_range(_, range_):
    """Resolve any expression to the given range."""
    return range_


class _Decred(int):
    """Placeholder objects which cannot be created by users.

    Instances of this class can be used as placeholders ensure that it is not
    something created by the user with special meaning.  It subclasses int so
    that it can be used in places where ordering and equality comparison is
    needed.
    """
    __slots__ = []
