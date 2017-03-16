"""The main drudge and tensor class definition."""

import contextlib
import functools
import inspect
import itertools
import operator
import types
import typing
import warnings
from collections.abc import Iterable, Sequence

from IPython.display import Math, display
from pyspark import RDD, SparkContext
from sympy import IndexedBase, Symbol, Indexed, Integer, Wild, latex

from .canonpy import Perm, Group
from .report import Report
from .term import (
    Range, sum_term, Term, parse_term, Vec, subst_factor_in_term,
    subst_vec_in_term, parse_terms, einst_term, diff_term, try_resolve_range
)
from .utils import ensure_symb, BCastVar, nest_bind, prod_


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
        return self._terms.map(lambda x: x.is_scalar).reduce(operator.and_)

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
        return self._terms.map(lambda x: x.has_base(base)).reduce(operator.or_)

    #
    # Printing support
    #

    def __str__(self):
        """Get the string representation of the tensor.

        Note that this function will **gather** all terms into the driver.

        """
        if self.n_terms == 0:
            return '0'
        else:
            return '\n + '.join(str(i) for i in self.local_terms)

    def latex(self, sep_lines=False):
        r"""Get the latex form for the tensor.

        The actual printing is dispatched to the drudge object for the
        convenience of tuning the appearance.

        Parameters
        ----------

        sep_lines : bool
            If terms should be put into separate lines by separating them with
            ``\\``.

        """
        return self._drudge.format_latex(self, sep_lines=sep_lines)

    def display(self, if_return=True, sep_lines=False):
        """Display the tensor in interactive IPython notebook sessions.

        Parameters
        ----------

        if_return
            If the resulted equation be returned rather than directly displayed.
            It can be disabled for displaying equation in the middle of a
            Jupyter cell.

        sep_lines
            If terms should be written into separate lines.

        """

        form = Math(self.latex(sep_lines=sep_lines))
        if if_return:
            return form
        else:
            display(form)
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

    def sort(self):
        """Sort the terms in the tensor.

        The terms will generally be sorted according to increasing complexity.
        """
        return self.apply(self._sort)

    @staticmethod
    def _sort(terms: RDD):
        """Sort the terms in the tensor."""
        return terms.sortBy(lambda term: term.sort_key)

    def merge(self):
        """Merge terms with the same vector and summation part.

        This function merges terms only when their summation list and vector
        part are *syntactically* the same.  So it is more useful when the
        canonicalization has been performed and the dummies reset.
        """

        # All the traits could be invalidated by merging.
        return Tensor(
            self._drudge, self._merge(self._terms)
        )

    def _merge(self, terms):
        """Get the term when they are attempted to be merged."""
        if not self._drudge.simple_merge:
            return terms.map(
                lambda term: ((term.sums, term.vecs), term.amp)
            ).reduceByKey(operator.add).map(
                lambda x: Term(x[0][0], x[1], x[0][1])
            )
        else:
            return terms.map(_decompose_term).reduceByKey(operator.add).map(
                lambda x: Term(x[0][0], x[1] * x[0][2], x[0][1])
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

        This is the master driver function for tensor simplification.

        """

        return Tensor(
            self._drudge, self._simplify(self._terms), expanded=True
        )

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

        # Simplify things like zero or deltas.
        if self._drudge.full_simplify:
            terms = self._simplify_amps(terms)
        terms = self._simplify_deltas(terms, False)

        # Canonicalize the terms and see if they can be merged.
        terms = self._canon(terms, True)
        # In rare cases, normal order could make the result unexpanded.
        #
        # TODO: Find a design to skip repartition in most cases.

        free_vars = self._get_free_vars(terms)
        terms = self._reset_dumms(terms, excl=free_vars)
        terms = self._merge(terms)

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
        return self._add(other * Integer(-1))

    def __rsub__(self, other):
        """Subtract the tensor from another quantity."""
        return (self * Integer(-1))._add(other)

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

    #
    # Substitution
    #

    def subst(self, lhs, rhs, wilds=None):
        """Substitute the all appearance of the defined tensor.

        When the given LHS is a plain SymPy symbol, all its appearances in the
        amplitude of the tensor will be replaced.  Or the LHS can also be
        indexed SymPy expression or indexed Vector, for which all of the
        appearances of the indexed base or vector base will be attempted to be
        matched against the indices on the LHS.  When a matching succeeds for
        all the indices, the RHS, with the substitution found in the matching
        performed, will be replace the indexed base in the amplitude, or the
        vector.  Note that for scalar LHS, the RHS must contain no vector.

        Since we do not commonly define tensors with wild symbols, an option
        ``wilds`` can be used to give a mapping translating plain symbols on the
        LHS and the RHS to the wild symbols that would like to be used.  The
        default value of None could make all **plain** symbols in the indices of
        the LHS to be translated into a wild symbol with the same name and no
        exclusion. And empty dictionary can be used to disable all such
        automatic translation.  The default value of None should satisfy most
        needs.

        """

        if isinstance(lhs, (Vec, Indexed)):
            base = lhs.base
        elif isinstance(lhs, Symbol):
            base = lhs
        else:
            raise TypeError(
                'Invalid LHS for substitution', lhs,
                'expecting vector, indexed, or symbol'
            )

        if not self.has_base(base):
            return self

        # We need to gather, and later broadcast all the terms.  The rational is
        # that the RHS is usually small in real problems.
        if isinstance(rhs, Tensor):
            rhs_terms = rhs.local_terms
        else:
            rhs_terms = parse_terms(rhs)

        if isinstance(lhs, (Symbol, Indexed)) and not all(
                i.is_scalar for i in rhs_terms
        ):
            raise ValueError('Invalid RHS for substituting a scalar', rhs)

        if wilds is None:
            if isinstance(lhs, (Indexed, Vec)):
                wilds = {
                    i: Wild(i.name) for i in lhs.indices if
                    isinstance(i, Symbol)
                    }
            else:
                wilds = {}

        if isinstance(lhs, Indexed):
            lhs = lhs.xreplace(wilds)
        elif isinstance(lhs, Vec):
            lhs = lhs.map(lambda x: x.xreplace(wilds))

        rhs_terms = [j.subst(wilds) for i in rhs_terms for j in i.expand()]

        expanded = self.expand()
        return expanded._subst(lhs, rhs_terms)

    def _subst(self, lhs: typing.Union[Vec, Indexed, Symbol], rhs_terms):
        """Core substitution function.

        This function assumes the self and the substituting terms are already
        fully expanded.  And the LHS and the RHS of the substitution have been
        replaced with the wilds if it is needed.
        """

        free_vars_local = (
            self.free_vars | set.union(*[i.free_vars for i in rhs_terms])
        )
        free_vars = self._drudge.ctx.broadcast(free_vars_local)
        dumms = self._drudge.dumms

        # We keep the dummbegs dictionary for each term and substitute all
        # appearances of the lhs one-by-one.

        subs_states = self._terms.map(lambda x: x.reset_dumms(
            dumms=dumms.value, excl=free_vars.value
        ))

        rhs_terms = self._drudge.ctx.broadcast(rhs_terms)

        if isinstance(lhs, (Indexed, Symbol)):
            res = nest_bind(subs_states, lambda x: subst_factor_in_term(
                x[0], lhs, rhs_terms.value,
                dumms=dumms.value, dummbegs=x[1], excl=free_vars.value
            ))
        else:
            res = nest_bind(subs_states, lambda x: subst_vec_in_term(
                x[0], lhs, rhs_terms.value,
                dumms=dumms.value, dummbegs=x[1], excl=free_vars.value
            ))

        res_terms = res.map(operator.itemgetter(0))
        return Tensor(
            self._drudge, res_terms, free_vars=free_vars_local, expanded=True
        )

    def subst_all(self, defs, simplify=False):
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
                res = i.act(res)
            elif isinstance(i, Sequence) and len(i) == 2:
                res = res.subst(i[0], i[1])
            else:
                raise TypeError(
                    'Invalid substitution', i,
                    'expecting definition or LHS/RHS pair'
                )
            if simplify:
                res = res.simplify().repartition()

        return res

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
    # Term filter and cherry picking
    #

    def filter(self, crit):
        """Filter out terms satisfying the given criterion."""
        return self.apply(
            lambda terms: terms.filter(crit),
            free_vars=None, repartitioned=False
        )

    #
    # Advanced manipulations.
    #

    def map2scalars(self, action, skip_vecs=False):
        """Map the given action to the scalars in the tensor.

        The given action should return SymPy expressions for SymPy expressions,
        the amplitude for each terms and the indices to the vectors, in the
        tensor.  Note that this function does not change the summations in the
        terms and the dummies.

        Parameters
        ----------

        action
            The callable to be applied to the scalars inside the tensor.

        skip_vecs
            When it is set, the callable will no longer be mapped to the indices
            to the vectors.  It could be used to boost the performance when we
            know that the action need no application on the indices.

        """

        return Tensor(self._drudge, self._terms.map(
            lambda x: x.map(action, skip_vecs=skip_vecs)
        ))

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


class TensorDef:
    """Definition of a tensor.
    """

    __slots__ = [
        '_tensor',
        '_base',
        '_exts',
        '_is_scalar'
    ]

    def __init__(self, base, exts, tensor: Tensor):
        """Initialize the tensor definition.

        In the same way as the initializer for the :py:class:`Tensor` class,
        this initializer is also unlikely to be used directly in user code.
        Drudge methods :py:meth:`Drudge.define` and
        :py:meth:`Drudge.define_einst` can be more convenient.
        """

        if isinstance(base, Vec):
            self._base = base
            self._is_scalar = False
        elif isinstance(base, (IndexedBase, Symbol)):
            self._base = base
            self._is_scalar = True
        else:
            raise TypeError(
                'Invalid base for tensor definition', base,
                'expecting vector or scalar base'
            )

        if isinstance(tensor, Tensor):
            self._tensor = tensor
        else:
            raise TypeError(
                'Invalid LHS for tensor definition', tensor,
                'expecting a tensor instance'
            )

        self._exts = []
        for i in exts:
            valid_ext = (
                isinstance(i, Sequence) and len(i) == 2 and
                isinstance(i[0], Symbol) and isinstance(i[1], Range)
            )
            if valid_ext:
                self._exts.append(tuple(i))
            else:
                raise TypeError(
                    'Invalid external index', i,
                    'expecting dummy and range pair'
                )
            continue

        # Additional processing for scalar replacement.
        if self._is_scalar:
            if not self._tensor.is_scalar:
                raise ValueError(
                    'Invalid tensor', self._tensor, 'for base', self._base,
                    'expecting a scalar'
                )
            if len(self._exts) == 0 and isinstance(self._base, IndexedBase):
                self._base = self._base.label

    #
    # Basic properties.
    #

    @property
    def is_scalar(self):
        """If the tensor defined is a scalar."""
        return self._is_scalar

    @property
    def rhs(self):
        """Get the right-hand-side of the definition."""
        return self._tensor

    @property
    def rhs_terms(self):
        """Gather the terms on the right-hand-side of the definition."""
        return self._tensor.local_terms

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

        Due to the scarcity of the usefulness of keeping both the unsimplified
        and the simplified tensor definition, this method will mutate the tensor
        with its simplified form.
        """

        self._tensor = self._tensor.simplify()
        return self

    def __eq__(self, other):
        """Compare two tensor definitions for equality.

        Note that similar to the equality comparison of tensors, here we only
        compare the syntactic equality rather than the mathematical equality.
        """

        if not isinstance(other, TensorDef):
            return False

        return self.lhs == other.lhs and self.rhs == other.rhs

    #
    # Representations.
    #

    def __str__(self):
        """Form simple readable string for a definition.
        """

        return ' = '.join([str(self.lhs), str(self.rhs)])

    def latex(self, sep_lines=False):
        r"""Get the latex form for the tensor definition.

        The result will just be the form from :py:meth:`Tensor.latex` with the
        RHS prepended.

        Parameters
        ----------

        sep_lines : bool
            If terms should be put into separate lines by separating them with
            ``\\``.

        """
        return self._tensor.drudge.format_latex(
            self, sep_lines=sep_lines
        )

    def display(self, if_return=True, sep_lines=False):
        """Display the tensor definition in interactive notebook sessions.

        The parameters here all have the same meaning as in
        :py:meth:`Tensor.display`.
        """

        form = Math(self.latex(sep_lines=sep_lines))
        if if_return:
            return form
        else:
            display(form)
            return

    #
    # Substitution.
    #

    def act(self, tensor, wilds=None):
        """Act the definition on a tensor.

        This method is the active voice version of the :py:meth:`Tensor.subst`
        function.  All appearances of the defined object in the tensor will be
        substituted.

        """

        if not isinstance(tensor, Tensor):
            tensor = self.rhs.drudge.sum(tensor)

        return tensor.subst(self.lhs, self.rhs, wilds=wilds)

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

        self._dumms = BCastVar(self._ctx, {})
        self._symms = BCastVar(self._ctx, {})
        self._resolvers = BCastVar(self._ctx, [])

        self._names = types.SimpleNamespace()

        self._tensor_methods = {}

    @property
    def ctx(self):
        """The Spark context of the drudge.
        """
        return self._ctx

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

    #
    # Name archive utilities.
    #

    def set_name(self, *args, **kwargs):
        """Set the object into the name archive of the drudge.

        For positional arguments, the str form of the given label is going to be
        used for the name of the object.  For keyword arguments, the keyword
        will be used for the name.
        """

        to_set = itertools.chain(
            ((str(i), i) for i in args), kwargs.items()
        )

        for label, obj in to_set:
            setattr(self._names, label, obj)
            continue

        return

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
            set.

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

            group = Group(gens)

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
            self.set_name(**{str(base.label): base})

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

    def add_resolver_for_dumms(self):
        """Add the resolver for the dummies for each range.

        With this method, the default dummies for each range will be resolved to
        be within the range for all of them.  This method should normally be
        called by all subclasses after the dummies for all ranges have been
        properly set.

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

        Note that this function can also be called on existing tensor objects
        with the same semantics on the terms.  Existing summations are not
        touched by it.

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

    def einst(self, summand) -> Tensor:
        """Create a tensor from Einstein summation convention.

        By calling this function, summations according to the Einstein summation
        convention will be added to the terms.  Note that for a symbol to be
        recognized as a summation, it must appear exactly twice in its
        **original form** in indices, and its range needs to be able to be
        resolved.  When a symbol is suspiciously an Einstein summation dummy but
        does not satisfy the requirement precisely, it will **not** be added as
        a summation, but a warning will also be given for reference.

        Note that in addition to creating tensors from scratch, this method can
        also be called on an existing tensor to add new summations.  In that
        case, no existing summations will be touched.
        """

        resolvers = self.resolvers
        if isinstance(summand, Tensor):
            summand_terms = summand.expand().terms
            return Tensor(self, summand_terms.map(
                lambda x: einst_term(x, resolvers.value)
            ), expanded=True)
        else:
            # We need to expand the possibly parenthesized user input.
            summand_terms = []
            for i in parse_terms(summand):
                summand_terms.extend(i.expand())

            return self.create_tensor(
                [einst_term(i, resolvers.value) for i in summand_terms]
            )

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
            all the indices being plain symbols whose range is able to be
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

    def define_einst(self, *args) -> TensorDef:
        """Make a tensor definition based on Einstein summation convention.

        Basically the same function as the :py:meth:`define`, just the content
        will be interpreted according to the Einstein summation convention.
        """

        if len(args) == 0:
            raise ValueError('Expecting arguments for definition.')

        base, exts = self._parse_def_lhs(args[:-1])
        tensor = self.einst(args[-1])
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
                exts = []
                for i in arg.indices:
                    range_ = try_resolve_range(i, {}, self.resolvers.value)
                    if range_ is None:
                        raise ValueError(
                            'Invalid index', i, 'in', args,
                            'range cannot be resolved'
                        )
                    exts.append((i, range_))
                    continue
                return base, exts
            else:
                return arg, ()
        else:
            return args[0], args[1:]

    #
    # Printing
    #

    def format_latex(self, inp, sep_lines=False):
        """Get the LaTeX form of a given tensor or tensor definition.

        Subclasses should fine-tune the appearance of the resulted LaTeX form by
        overriding methods ``_latex_sympy``, ``_latex_vec``, and
        ``_latex_vec_mul``.

        """

        if isinstance(inp, Tensor):

            n_terms = inp.n_terms
            inp_terms = inp.local_terms
            prefix = ''

        elif isinstance(inp, TensorDef):

            n_terms = inp.rhs.n_terms
            inp_terms = inp.rhs_terms
            prefix = (
                         self._latex_sympy(inp.lhs) if inp.is_scalar else
                         self._latex_vec(inp.lhs)
                     ) + ' = '

        else:
            raise TypeError('Invalid object to form into LaTeX.')

        if n_terms == 0:
            return prefix + '0'

        terms = []
        for i, v in enumerate(inp_terms):
            term = self._latex_term(v)
            if i != 0 and term[0] not in {'+', '-'}:
                term = ' + ' + term
            terms.append(term)
            continue

        term_sep = r' \\ ' if sep_lines else ' '

        return prefix + term_sep.join(terms)

    def _latex_term(self, term):
        """Format a term into LaTeX form.

        This method does not generally need to be overridden.
        """

        parts = []

        factors, coeff = term.amp_factors
        if coeff == 1:
            coeff_latex = None
        elif coeff == -1:
            parts.append('-')
            coeff_latex = None
        else:
            coeff_latex = self._latex_sympy(coeff)
            if coeff_latex[0] == '-':
                parts.append('-')
                coeff_latex = coeff_latex[1:]

        parts.extend(r'\sum_{{{} \in {}}}'.format(
            i, j.label
        ) for i, j in term.sums)

        if coeff_latex is not None:
            parts.append(coeff_latex)

        if len(factors) > 0:
            parts.extend(self._latex_sympy(i) for i in factors)

        vecs = self._latex_vec_mul.join(
            self._latex_vec(i) for i in term.vecs
        )
        parts.append(vecs)

        return ' '.join(parts)

    @staticmethod
    def _latex_sympy(expr):
        """Get the LaTeX form of SymPy expressions.

        The default SymPy method will be used, subclasses can override this
        method for fine tuning of the form.
        """
        return latex(expr)

    def _latex_vec(self, vec):
        """Get the LaTeX form of a vector.

        By default, the vector name is going to be put into boldface, and the
        indices are put into the subscripts.
        """

        head = r'\mathbf{{{}}}'.format(vec.label)
        indices = ', '.join(self._latex_sympy(i) for i in vec.indices)
        return r'{}_{{{}}}'.format(head, indices)

    _latex_vec_mul = r' \otimes '

    @contextlib.contextmanager
    def report(self, filename, title):
        """Make a report for results.

        This function should be used within a ``with`` statement to open a
        report for results.

        """

        report = Report(filename, title)
        yield report
        report.write()


#
# Small static utilities
#


def _union(orig, new):
    """Union the two sets and return the first."""
    orig |= new
    return orig


def _decompose_term(term):
    """Decompose a term for simple merging.

    The given term will be decomposed into a pair, where the first field has the
    summations, vectors, and product of factors containing at least one dummy.
    And the second field contains factors involving no dummies.
    """

    factors, coeff = term.amp_factors
    return (
        (term.sums, term.vecs, prod_(factors)),
        coeff
    )


def _is_nonzero(term):
    """Test if a term is trivially non-zero."""
    return term.amp != 0
