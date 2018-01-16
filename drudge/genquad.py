"""Base drudge for general quadratic algebra."""

import abc
import collections
import functools
import itertools
import operator
import typing

from pyspark import RDD
from sympy import sympify, Expr, Integer, KroneckerDelta, SympifyError

from .drudge import Drudge
from .term import Term, Vec, parse_terms, ATerms, Terms
from .utils import nest_bind, sympy_key


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


#
# Utility class for common problems.
#

class GenQuadLatticeDrudge(GenQuadDrudge):
    r"""Drudge for general quadratic algebra with concentration on the bases.

    The name of this drudge class can be slightly misleading.  This class is
    designed for algebraic systems where the generators can optionally be
    indexed by indices called the lattice indices.  Generators at different
    lattice sites are commutative, and the normal-ordering and commutation for
    generators with the same lattice index are solely determined by the base.

    Parameters
    ----------

    ctx
        The spark context.

    order
        An iterable giving the order of the vector bases in the normal order.
        Earlier vectors will be attempted to be put into earlier positions.

    comms
        The commutators, given as a mapping with keys being pairs of vector
        bases.  The values should give information about the commutation, which
        can be anything interpretable as a sum of terms giving the value of the
        commutator, or it can also be a pair with the second entry giving the
        phase of the commutation, which by default is unity.  For an entry::

            (a, b): (kappa, phi)

        we got the commutation rule

        .. math::

            [a, b] = \phi b a + \kappa

        The phase has to be a scalar quantity.  When the vectors inside the
        commutator have no index, they are going to be indexed by the indices
        from the vectors being commuted during the normal-ordering operation.

        The values can also be callable objects, which is going to be called
        with the two vectors to commute.  The return value have the same
        semantics as above, just they can be computed on the fly, rather than
        begin static for all lattice indices.  Note that this does not change
        the semantics of lattice algebra, different generators with different
        lattice indices always commute.

    assume_comm

        If vectors with their commutator unspecified be assumed to be
        commutative. It is false by default.

    kwargs
        All the rest of the parameters are given to be base class.

    """

    def __init__(
            self, ctx, order: typing.Iterable[Vec],
            comms: typing.Mapping[typing.Tuple[Vec, Vec], typing.Union[
                ATerms, typing.Tuple[ATerms, Expr]
            ]], assume_comm=False, **kwargs):
        """Initialize the drudge object."""
        super().__init__(ctx, **kwargs)
        self._swapper = self._form_swapper(order, comms, assume_comm)

    @property
    def swapper(self):
        """The swapper based on the given order and commutators."""
        return self._swapper

    def _form_swapper(self, order, comms_inp, assume_comm):
        """Form the swapper based on the input."""

        #
        # Normalization of the order and commutators.
        #

        base_ranks = {}
        for i, v in enumerate(order):
            if v in base_ranks:
                raise ValueError(
                    'Duplicated generator in the normal order', v
                )
            base_ranks[v] = i
            continue

        comms = {}
        for k, v in comms_inp.items():
            invalid_key = (
                    not isinstance(k, typing.Sequence) or len(k) != 2
                    or any(not isinstance(i, Vec) for i in k)
            )
            if invalid_key:
                raise ValueError(
                    'Invalid bases to commute, expecting two vectors', k
                )

            if isinstance(v, typing.Sequence):
                if len(v) != 2:
                    raise ValueError(
                        'Invalid commutator, commutator and phase expected', v
                    )
                comm, phase = v
            else:
                comm = v
                phase = _UNITY

            try:
                phase = sympify(phase)
            except SympifyError as exc:
                raise ValueError(
                    'Nonsympifiable phase of commutation', phase, exc
                )

            if not callable(comm):
                try:
                    comm = parse_terms(comm)
                except Exception as exc:
                    raise ValueError(
                        'Invalid commutator result', comm, exc
                    )

            comms[(k[0], k[1])] = (comm, phase)
            continue

        swap_info = _SwapInfo(
            base_ranks=base_ranks, comms=comms, assume_comm=assume_comm
        )

        bcast_swap_info = self.ctx.broadcast(swap_info)

        return functools.partial(
            _swap_lattice_gens, bcast_swap_info=bcast_swap_info
        )


_SwapInfo = collections.namedtuple('_SwapInfo', [
    'base_ranks',
    'comms',
    'assume_comm'
])


def _swap_lattice_gens(vec1: Vec, vec2: Vec, bcast_swap_info):
    """Swap the generators with lattice indices."""
    swap_info: _SwapInfo = bcast_swap_info.value
    base_ranks = swap_info.base_ranks
    comms = swap_info.comms
    assume_comm = swap_info.assume_comm
    vecs = (vec1, vec2)

    indices1, indices2 = [i.indices for i in vecs]
    if len(indices1) != len(indices2):
        raise ValueError('Unmatching lattice indices for', vec1, vec2)

    base1, base2 = [vec.base for vec in vecs]
    try:
        rank1 = base_ranks[base1]
        rank2 = base_ranks[base2]
    except KeyError as exc:
        raise ValueError(
            'Vector with unspecified normal order', exc.args
        )

    if rank1 < rank2:
        return None
    elif rank1 == rank2:
        # Same generator at possibly different sites always commute.
        key1, key2 = [
            tuple(sympy_key(j) for j in i) for i in (indices1, indices2)
        ]
        if key1 <= key2:
            return None
        else:
            return _UNITY, _NOUGHT
    else:
        given = (base1, base2)
        rev = (base2, base1)
        if_rev = False

        if given in comms:
            comm, phase = comms[given]
            comm_factor = _UNITY
        elif rev in comms:
            # a b = phi b a + kappa => phi b a = a b - kappa
            # => b a = (a b - kappa) / phi
            comm, phase = comms[rev]
            comm_factor = -1 / phase
            phase = 1 / phase
            if_rev = True
        elif assume_comm:
            comm = _NOUGHT
            phase = _UNITY
            comm_factor = _UNITY
        else:
            raise ValueError(
                'Commutation rules unspecified', vec1, vec2
            )

        if callable(comm):
            try:
                comm = parse_terms(
                    comm(vec2, vec1) if if_rev else comm(vec1, vec2)
                )
            except Exception as exc:
                raise ValueError(
                    'Invalid commutator', comm, exc
                )

        delta = functools.reduce(operator.mul, (
            KroneckerDelta(i, j) for i, j in zip(indices1, indices2)
        ), _UNITY)

        terms = Terms(
            Term(term.sums, delta * comm_factor * term.amp, tuple(
                i[indices1] if len(i.indices) == 0 else i
                for i in term.vecs
            ))
            for term in comm
        )
        return phase, terms


# Small utility constants.

_UNITY = Integer(1)
_NOUGHT = Integer(0)
