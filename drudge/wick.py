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
from .term import Term, Vec
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
        return terms.flatMap(lambda term: wick_expand(
            term, comparator=comparator, contractor=contractor, phase=phase,
            symms=symms.value
        ))


#
# Utility functions.
#

def wick_expand(term: Term, comparator, contractor, phase, symms=None):
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
        contrs = _get_all_contrs(vecs, contractor, term)
        vec_order = list(range(n_vecs))
    else:
        term = _preproc_term(term, symms)
        vecs = term.vecs
        vec_order, contrs = _sort_vecs(vecs, comparator, contractor, term)

    expander = _WickExpander(vecs, vec_order, contrs, phase, contr_all)
    expanded = expander.expand(term.amp)

    return [Term(term.sums, i[0], i[1]) for i in expanded]


def _sort_vecs(vecs, comparator, contractor, term):
    """Sort the vectors and get the contraction values.

    Here insertion sort is used to sort the vectors into the normal order
    required by the comparator.
    """

    n_vecs = len(vecs)
    contrs = [{} for _ in range(n_vecs)]

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

            contr_val = contractor(prev_vec, pivot_vec, term)
            if contr_val != 0:
                contrs[prev_i][pivot_i] = contr_val
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


def _get_all_contrs(vecs, contractor, term):
    """Generate all possible contractions.

    This function is going to be called when we do not actually need to normal
    order the vectors and only need the results where all the vectors are
    contracted.
    """
    n_vecs = len(vecs)
    contrs = []

    for i in range(n_vecs):
        curr_contrs = {}
        for j in range(i, n_vecs):
            vec_prev = vecs[i]
            vec_lat = vecs[j]
            contr_val = contractor(vec_prev, vec_lat, term)
            if contr_val != 0:
                curr_contrs[j] = contr_val
            continue
        contrs.append(curr_contrs)
        continue

    return contrs


class _WickExpander:
    """Expander of vectors based on Wick theorem.

    Here, the problem should be specified by a sequence ``vecs``, where each
    entry should be a pair of a unique integral identifier for the vector and
    the actual vector.  The given contractions will be queried with the
    identifier of the first vector for a dictionary of non-vanishing
    contractions with vectors **later** in the sequence.
    """

    def __init__(self, vecs, vec_order, contrs, phase, contr_all):
        """Initialize the expander."""

        self.vecs = vecs
        self.vec_order = vec_order
        self.contrs = contrs
        self.phase = phase
        self.contr_all = contr_all
        self.n_vecs = len(self.vecs)

    def expand(self, base_amp):
        """Make the Wick expansion."""

        expanded = []
        avail = [True for _ in range(self.n_vecs)]
        self._add_terms(expanded, base_amp, avail, 0)
        return expanded

    def _add_terms(self, expanded, amp, avail, pivot):
        """Add terms recursively."""

        vecs = self.vecs
        n_vecs = len(vecs)
        vec_order = self.vec_order
        contrs = self.contrs
        contr_all = self.contr_all
        phase = self.phase

        # Find the actual pivot, which has to be available.
        try:
            pivot = next(i for i in range(pivot, n_vecs - 1) if avail[i])
        except StopIteration:
            # When everything is already decided, add the current term.
            if not contr_all or all(not i for i in avail):
                rem_idxes = [i for i in vec_order if avail[i]]
                final_phase = _get_perm_phase(rem_idxes, phase)
                expanded.append((
                    final_phase * amp, tuple(vecs[i] for i in rem_idxes)
                ))
            return

        pivot_contrs = contrs[pivot]
        if contr_all and len(pivot_contrs) == 0:
            return

        if not contr_all:
            self._add_terms(expanded, amp, avail, pivot + 1)

        avail[pivot] = False
        n_vecs_between = 0
        for vec_idx in range(pivot + 1, self.n_vecs):
            if avail[vec_idx]:
                if vec_idx in pivot_contrs:
                    avail[vec_idx] = False
                    contr_val = pivot_contrs[vec_idx]
                    contr_phase = phase ** n_vecs_between
                    self._add_terms(
                        expanded, amp * contr_val * contr_phase,
                        avail, pivot + 1
                    )
                    avail[vec_idx] = True
                n_vecs_between += 1
            continue

        avail[pivot] = True
        return


def _get_perm_phase(order, phase):
    """Get the phase of the given permutation of points."""
    n_points = len(order)
    return phase ** sum(
        1 for i in range(n_points) for j in range(i + 1, n_points)
        if order[i] > order[j]
    )
