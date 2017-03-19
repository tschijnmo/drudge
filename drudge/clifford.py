"""Drudges for clifford algebra."""

import functools
import itertools
import operator
import typing

from pyspark import RDD
from sympy import Expr, Integer, KroneckerDelta

from .term import Vec, Term
from .wick import WickDrudge


def inner_by_delta(vec1: Vec, vec2: Vec):
    """Compute the inner product of two vectors by delta.

    The two vectors are assumed to be from the same base and have the same
    number of indices, or ValueError will be raised.
    """

    indices1 = vec1.indices
    indices2 = vec2.indices
    if vec1.label != vec2.label or len(indices1) != len(indices2):
        raise ValueError(
            'Invalid vectors to computer inner product by delta', (vec1, vec2)
        )

    return functools.reduce(operator.mul, (
        KroneckerDelta(i, j) for i, j in zip(indices1, indices2)
    ), Integer(1))


class CliffordDrudge(WickDrudge):
    r"""Drudge for Clifford algebras.

    A Clifford algebra over a inner product space :math:`V` is an algebraic
    system with

    .. math::

        uv + vu = 2 \langle u, v \rangle

    for all :math:`u, v \in V`.

    This drudge should work for any Clifford algebra with given inner product
    function.

    """

    Inner = typing.Callable[[Vec, Vec], Expr]

    def __init__(self, ctx, inner: Inner = inner_by_delta, **kwargs):
        """Initialize the drudge.

        Parameters
        ----------

        ctx
            The context for Spark.

        inner
            The callable to compute the inner product of two vectors.  By
            default, the inner product of vectors of the same base and the same
            number of indices will be computed to be the delta, or
            ``ValueError`` will be raised.

        kwargs
            All other keyword arguments will be forwarded to the base class
            :py:class:`WickDrudge`.

        """

        super().__init__(ctx, **kwargs)

        self._inner = inner
        self._contractor = functools.partial(
            _contract4clifford, inner=inner
        )
        self._collapse = functools.partial(
            _collapse4clifford, inner=inner
        )

    @property
    def phase(self):
        """The phase for Clifford algebra, negative unity."""
        return Integer(-1)

    @property
    def comparator(self):
        """Comparator for Clifford algebra.

        Here we just compare vectors by the default sort key for vectors.
        """
        return _compare_by_sort_key

    @property
    def contractor(self):
        """Contractor for Clifford algebra.

        The inner product function will be invoked.
        """
        return self._contractor

    def normal_order(self, terms: RDD, **kwargs):
        """Put vectors in Clifford algebra in normal-order.

        After the normal-ordering by Wick expansion, adjacent equal vectors will
        be collapsed by rules of Clifford algebra.
        """

        res = super().normal_order(terms, **kwargs)
        return res.map(self._collapse)


def _compare_by_sort_key(vec1: Vec, vec2: Vec, _: Term):
    """Compare the two vectors just by their sort key."""
    return vec1.sort_key > vec2.sort_key


def _contract4clifford(
        vec1: Vec, vec2: Vec, _: Term, *,
        inner
):
    """Contract two vectors by Clifford rules."""
    return inner(vec1, vec2) * Integer(2)


def _collapse4clifford(term: Term, *, inner):
    """Collapse adjacent equal vectors by Clifford algebras."""

    vecs = []
    amp = term.amp
    for k, g in itertools.groupby(term.vecs):

        n_vecs = sum(1 for _ in g)
        n_inners = n_vecs // 2
        n_rem = n_vecs % 2

        # Skip inner product computation when it is not needed.
        if n_inners > 0:
            amp *= inner(k, k) ** n_inners

        if n_rem > 0:
            vecs.append(k)

        continue

    return Term(term.sums, amp, tuple(vecs))
