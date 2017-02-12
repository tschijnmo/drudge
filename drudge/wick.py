"""Drudge for wick-style algebraic systems.

In this module, we have a abstract base class for Wick-style algebraic systems,
as well as function helpful for its subclasses.
"""

import abc
import collections
import typing

from pyspark import RDD
from sympy import Expr

from .drudge import Drudge
from .term import Term, Vec, simplify_deltas_in_expr, compose_simplified_delta
from .utils import sympy_key


class WickDrudge(Drudge, abc.ABC):
    """Drudge for Wick-style algebras.

    A Wick-style algebra is an algebraic system where the commutator between any
    generators of the algebra is a simple scalar value.  This drudge will
    attempt to put the vectors into normal order based on the given comparator
    and contractor by Wick theorem.

    """

    @abc.abstractproperty
    def contractor(self) -> typing.Callable[[Vec, Vec, Term], Expr]:
        """Get the contractor for the algebraic system.

        The contractor is going to be called with two vectors to return the
        value of their contraction.

        """
        pass

    @abc.abstractproperty
    def phase(self):
        """Get the phase for the commutation rule.

        The phase should be a constant defining the phase of the commutation
        rule.
        """
        pass

    @abc.abstractproperty
    def comparator(self) -> typing.Callable[[Vec, Vec, Term], bool]:
        """Get the comparator for the canonicalized vectors.

        The normal ordering operation will be performed according to this
        comparator.  It will be called with two **canonicalized vectors** for a
        boolean value.  True should be returned if the first given vector is
        less than the second vector.  The two vectors will be attempted to be
        transposed when False is returned.

        """
        pass

    def normal_order(self, terms: RDD):
        """Normal order the terms according to generalized Wick theorem.

        The actual expansion is based on the information given in the subclasses
        by the abstract properties.

        """
        comparator = self.comparator
        contractor = self.contractor
        phase = self.phase
        symms = self.symms
        resolvers = self.resolvers

        return terms.flatMap(lambda term: wick_expand_term(
            term, comparator=comparator, contractor=contractor, phase=phase,
            symms=symms.value, resolvers=resolvers.value
        ))


#
# Utility functions.
#

def wick_expand_term(
        term: Term, comparator, contractor, phase, symms=None, resolvers=()
):
    """Expand a Term by wick theorem.

    When the comparator is None, it is assumed that only terms with all the
    vectors contracted will be kept, as in the case in vacuum expectation value
    evaluation in many-body physics.

    """

    symms = {} if symms is None else symms
    contr_all = comparator is None
    vecs = term.vecs
    n_vecs = len(vecs)

    if n_vecs == 0:
        return [term]
    elif n_vecs == 1:
        return [] if contr_all else [term]

    # First the vectors and contractions need to be formed as required by the
    # Wick expander.

    if contr_all:
        contrs = _get_all_contrs(term, contractor, resolvers=resolvers)
        vec_order = None
    else:
        term = _preproc_term(term, symms)
        vec_order, contrs = _sort_vecs(
            term, comparator, contractor, resolvers=resolvers
        )

    schemes = _compute_wick_schemes(vec_order, contrs)

    return [
        _form_term_from_wick(term, contrs, phase, resolvers, i)
        for i in schemes
        ]


def _sort_vecs(term, comparator, contractor, resolvers):
    """Sort the vectors and get the contraction values.

    Here insertion sort is used to sort the vectors into the normal order
    required by the comparator.
    """

    vecs = term.vecs
    n_vecs = len(vecs)
    contrs = [{} for _ in range(n_vecs)]
    sums_dict = term.dumms

    vec_order = list(range(0, n_vecs))

    front = 2
    pivot = 1
    while pivot < n_vecs:

        pivot_i = vec_order[pivot]
        pivot_vec = vecs[pivot_i]
        prev = pivot - 1

        if pivot == 0 or comparator(vecs[vec_order[prev]], pivot_vec, term):
            pivot, front = front, front + 1
        else:

            prev_i = vec_order[prev]
            prev_vec = vecs[prev_i]
            vec_order[prev], vec_order[pivot] = pivot_i, prev_i

            contr_amp, contr_substs = simplify_deltas_in_expr(
                sums_dict, contractor(prev_vec, pivot_vec, term), resolvers
            )
            if contr_amp != 0:
                contrs[prev_i][pivot_i] = (
                    contr_amp, tuple(contr_substs.items())
                )
            pivot -= 1

        continue

    return vec_order, contrs


def _preproc_term(term, symms):
    """Prepare the term for Wick expansion.

    This is the preparation task for normal ordering.  The term will be
    canonicalized with all the vectors considered the same.
    """

    # Make the dummies factory to canonicalize the vectors.
    dumms = collections.defaultdict(list)
    for i, v in term.sums:
        dumms[v].append(i)
    for i in dumms.values():
        i.sort(key=sympy_key)

    canon_term = (
        term.canon(symms=symms, vec_colour=lambda idx, vec, term: 0)
            .reset_dumms(dumms)[0]
    )

    return canon_term


def _get_all_contrs(term, contractor, resolvers):
    """Generate all possible contractions.

    This function is going to be called when we do not actually need to normal
    order the vectors and only need the results where all the vectors are
    contracted.
    """

    vecs = term.vecs
    n_vecs = len(vecs)
    contrs = []
    sums_dict = term.dumms

    for i in range(n_vecs):
        curr_contrs = {}
        for j in range(i, n_vecs):
            vec_prev = vecs[i]
            vec_lat = vecs[j]
            contr_amp, contr_substs = simplify_deltas_in_expr(
                sums_dict, contractor(vec_prev, vec_lat, term), resolvers
            )
            if contr_amp != 0:
                curr_contrs[j] = (contr_amp, tuple(contr_substs.items()))
            continue
        contrs.append(curr_contrs)
        continue

    return contrs


def _compute_wick_schemes(vec_order, contrs):
    """Compute all the Wick expansion schemes.

    The vector order should be a sequence giving indices of vectors.  When it is
    None, it means that all vectors needs to be contracted.  The contractions
    should be a sequence of hash maps giving the amplitude and substitution of
    each contraction.

    The expansion result is a list of pairs, with the first field holding the
    permutation of the given vectors for the contraction term, and the second
    being the number of vectors contracted. Adjacent pairs in the first section
    are are contracted, and the second section contains the remaining vectors
    ordered as in the given vector order.
    """

    schemes = []
    avail = [True for _ in contrs]
    _add_wick(schemes, avail, 0, [], vec_order, contrs)
    return schemes


def _add_wick(schemes, avail, pivot, contred, vec_order, contrs):
    """Add Wick expansion schemes recursively."""

    n_vecs = len(avail)
    contr_all = vec_order is None

    # Find the actual pivot, which has to be available.
    try:
        # Last vector can never be pivot.
        pivot = next(i for i in range(pivot, n_vecs - 1) if avail[i])
    except StopIteration:
        # When everything is already decided, add the current term.
        if not contr_all or all(not i for i in avail):
            vec_perm = list(contred)
            if not contr_all:
                vec_perm.extend(i for i in vec_order if avail[i])
            schemes.append((
                vec_perm, len(contred)
            ))
        return

    pivot_contrs = contrs[pivot]
    if contr_all and len(pivot_contrs) == 0:
        return

    if not contr_all:
        _add_wick(schemes, avail, pivot + 1, contred, vec_order, contrs)

    avail[pivot] = False
    for vec_idx in range(pivot + 1, n_vecs):
        if avail[vec_idx] and vec_idx in pivot_contrs:
            avail[vec_idx] = False
            contred.extend([pivot, vec_idx])
            _add_wick(
                schemes, avail, pivot + 1, contred, vec_order, contrs
            )
            avail[vec_idx] = True
            contred.pop()
            contred.pop()
        continue

    avail[pivot] = True
    return


def _form_term_from_wick(term, contrs, phase, resolvers, wick_res):
    """Generate a full Term from a Wick expansion scheme.
    """

    sums_dict = term.dumms

    perm, n_contred = wick_res
    phase = _get_perm_phase(perm, phase)

    amp = phase
    substs = {}
    for contr_i in range(0, n_contred, 2):
        contr_amp, contr_substs = contrs[perm[contr_i]][perm[contr_i + 1]]
        amp, _ = compose_simplified_delta(
            amp * contr_amp, contr_substs,
            substs, sums_dict=sums_dict, resolvers=resolvers
        )
        continue

    vecs = tuple(term.vecs[i] for i in perm[n_contred:])
    return term.subst(
        substs, amp=amp * term.amp, vecs=vecs, purge_sums=True
    )


def _get_perm_phase(order, phase):
    """Get the phase of the given permutation of points."""
    n_points = len(order)
    return phase ** sum(
        1 for i in range(n_points) for j in range(i + 1, n_points)
        if order[i] > order[j]
    )
