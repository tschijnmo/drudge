"""Base drudge for general quadratic algebra."""

import abc
import collections
import itertools
import typing

from pyspark import RDD
from sympy import sympify, Expr

from .drudge import Drudge
from .term import Term, Vec, parse_terms, ATerms
from .utils import nest_bind


class GenQuadDrudge(Drudge, abc.ABC):
    r"""Drudge for general quadratic algebra.

    This abstract base class encompasses a wide range of algebraic systems.  By
    a quadratic algebra, we mean any algebraic system with commutation rules

    .. math::

        a b = \phi b a + \kappa

    for any two elements :math:`a` and :math:`b` in the algebra with
    :math:`\phi` a scalar and :math:`\kappa` any element in the algebra.  This
    includes all Lie algebra systems by fixing :math:`\phi` to plus unity.
    Other algebra systems with other :math:`\phi` can also be treated as long as
    it is a scalar.

    For the special case of :math:`\phi = \pm 1` and :math:`\kappa` being a
    scalar, :py:class:`WickDrudge` should be used, which utilizes the special
    structure and has much better performance.

    """

    def __init__(self, ctx, full_balance=False, **kwargs):
        """Initialize the drudge."""

        super().__init__(ctx, **kwargs)
        self._full_balance = False  # For linter.
        self.full_balance = full_balance

    @property
    def full_balance(self) -> bool:
        """If full load-balancing is to be performed during normal-ordering.
        """
        return self._full_balance

    @full_balance.setter
    def full_balance(self, val: bool):
        """Set the full load-balancing option."""

        if not isinstance(val, bool):
            raise TypeError(
                'Invalid option for full balancing', val, 'expecting boolean'
            )
        self._full_balance = val

    Swapper = typing.Callable[
        [Vec, Vec], typing.Optional[typing.Tuple[Expr, ATerms]]
    ]

    @property
    @abc.abstractmethod
    def swapper(self) -> Swapper:
        """The function to be called with two operators to commute.

        It is going to be called with two vectors.  When they are already in
        desired order, a ``None`` should be returned.  Or the phase of the
        commutation should be returned as a SymPy expression, along with the
        commutator, which can be anything that can be interpreted as terms.

        It is named as ``swapper`` is avoid any confusion about the established
        meaning of the word ``commutator`` in mathematics.
        """
        pass

    def normal_order(self, terms: RDD, **kwargs):
        """Normal order the terms in the RDD."""

        if len(kwargs) > 0:
            raise ValueError('Invalid keyword arguments', kwargs)

        symms = self.symms
        swapper = self.swapper
        resolvers = self.resolvers

        init = terms.map(lambda term: _NOState(
            pivot=1, front=2, term=term.canon4normal(symms.value)
        ))

        res = nest_bind(init, lambda x: _sort_vec(
            x, swapper=swapper, resolvers=resolvers.value
        ), full_balance=self.full_balance)

        return res.map(lambda x: x.term)


#
# A state in the normal-ordering process.
#
# Basically this is a small variation of insertion sort.  Pivot is the index of
# the vector to swap, possibly.  Front is the index for the next pivot.
#

_NOState = collections.namedtuple('_NOState', [
    'pivot',
    'front',
    'term'
])


def _sort_vec(no_state: _NOState, swapper: GenQuadDrudge.Swapper, resolvers):
    """Perform one step in the insertion sorting process.

    This function is designed to work with the nest-bind protocol.
    """

    orig_term = no_state.term
    vecs = orig_term.vecs
    n_vecs = len(vecs)
    pivot = no_state.pivot

    if pivot >= n_vecs:
        # End of the outer loop.
        return None

    # To be used by the end of the inner loop.
    move2front = [_NOState(
        pivot=no_state.front, front=no_state.front + 1, term=no_state.term
    )]

    if pivot == 0:
        return move2front

    prev = pivot - 1
    swap_res = swapper(vecs[prev], vecs[pivot])
    if swap_res is None:
        return move2front

    phase = sympify(swap_res[0])
    comm_terms = []
    for i in parse_terms(swap_res[1]):
        comm_terms.extend(i.expand())

    # Now we need to do a swap.
    head = vecs[:prev]
    tail = vecs[pivot + 1:]
    res_states = []

    swapped_vecs = head + (vecs[pivot], vecs[prev]) + tail
    swapped_amp = phase * orig_term.amp
    if swapped_amp != 0:
        swapped_term = Term(
            orig_term.sums, swapped_amp, swapped_vecs
        )
        res_states.append(_NOState(
            pivot=prev, front=no_state.front, term=swapped_term
        ))

    for comm_term in comm_terms:
        if comm_term.amp == 0:
            continue

        comm_vecs = comm_term.vecs
        res_term = Term(orig_term.sums, comm_term.amp * orig_term.amp, tuple(
            itertools.chain(head, comm_vecs, tail)
        )).simplify_deltas(resolvers)
        if res_term.amp == 0:
            continue

        if len(comm_vecs) == 0:
            # Old front, new index.
            res_pivot = no_state.front - 2
        else:
            # Index of the first newly inserted vector.
            res_pivot = prev

        res_states.append(_NOState(
            pivot=res_pivot, front=res_pivot + 1, term=res_term
        ))
        continue

    return res_states
