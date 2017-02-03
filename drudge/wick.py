"""Drudge for wick-style algebraic systems.

In this module, we have a abstract base class for Wick-style algebraic systems,
as well as function helpful for its subclasses.
"""

import abc
import collections
import functools

from pyspark import RDD

from .canon import canon_factors
from .drudge import Drudge
from .term import Term
from .utils import sympy_key


class WickDrudge(Drudge, abc.ABC):
    """Drudge for Wick-style algebras.

    A Wick-style algebra is an algebraic system where the commutator between any
    generators of the algebra is a simple scalar value.  This drudge will
    attempt to put the vectors into normal order based on the given comparator
    and contractor by Wick theorem.

    """

    @abc.abstractproperty
    def contractor(self):
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
    def comparator(self):
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
        return terms.flatMap(functools.partial(
            wick_expand, comparator=self.comparator,
            contractor=self.contractor, phase=self.phase
        ))


#
# Utility functions.
#

def wick_expand(term: Term, comparator, contractor, phase):
    """Expand a Term by wick theorem.

    When the comparator is None, it is assumed that only terms with all the
    vectors contracted will be kept, as in the case in vacuum expectation value
    evaluation in many-body physics.

    """

    contr_all = comparator is None
    n_vecs = len(term.vecs)
    if n_vecs == 0:
        return [term]
    elif n_vecs == 1:
        return [] if contr_all else [term]

    # First the vectors and contractions need to be formed as required by the
    # Wick expander.

    if contr_all:
        vecs, contrs = _get_all_contrs(term.vecs, contractor)
        base_amp = term.amp
    else:
        vecs = _prepare_vecs(term)
        res_phase, vecs, contrs = _sort_vecs(
            vecs, comparator, contractor, phase
        )
        base_amp = term.amp * res_phase

    expander = _WickExpander(vecs, contrs, phase, contr_all)
    expanded = expander.expand(base_amp)

    return [Term(term.sums, i[0], i[1]) for i in expanded]


def _sort_vecs(vecs, comparator, contractor, phase):
    """Sort the vectors and get the contraction values.

    Here insertion sort is used to sort the vectors into the normal order
    required by the comparator.
    """

    n_vecs = len(vecs)
    contrs = [{} for _ in range(n_vecs)]
    res_phase = 1

    front = 2
    pivot = 1
    while pivot < n_vecs:

        pivot_vec = vecs[pivot]
        prev = pivot - 1
        prev_vec = vecs[prev]

        if pivot == 0 or comparator(prev_vec[2], pivot_vec[2]):
            pivot, front = front, front + 1
        else:
            vecs[pivot] = vecs[prev]
            vecs[prev] = pivot_vec
            res_phase *= phase

            contr_val = contractor(prev_vec[1], pivot_vec[1])
            if contr_val != 0:
                contrs[pivot_vec[0]][prev_vec[0]] = contr_val
            pivot -= 1

        continue

    return res_phase, vecs, contrs


def _prepare_vecs(term):
    """Prepare the vectors.

    This is the preparation task for normal ordering.  The vectors will be
    transformed into triples: original index, original vector, canonicalized
    vector.
    """

    # Make the dummies factory to canonicalize the vectors.
    dumms = collections.defaultdict(list)
    for i, v in term.sums:
        dumms[v].append(i)
    for i in dumms.values():
        i.sort(key=sympy_key)  # Not necessary, for ease of debugging.
    sums = term.sums

    symms = {}
    # Mock symmetry dictionary.
    #
    # TODO: Maybe make symmetries on vectors available here.
    #
    # Normally this should not be useful.  What we have here is sufficient.

    vecs = []  # The result.
    for idx, vec in enumerate(term.vecs):
        new_sums, factors, _ = canon_factors(sums, [(vec, 0)], symms)

        # The dummies are all original dummies, hence cannot conflict with free
        # variables.
        _, substs, _ = term.reset_sums(new_sums, dumms)
        canoned_vec = vec.map(lambda x: x.subs(substs, simultaneous=True))
        vecs.append((idx, vec, canoned_vec))

    return vecs


def _get_all_contrs(vecs, contractor):
    """Generate all possible contractions.

    This function is going to be called when we do not actually need to normal
    order the vectors and only need the results where all the vectors are
    contracted.
    """
    vecs = list(enumerate(vecs))
    contrs = []
    for vec in vecs:
        curr_contrs = {}
        for i, v in vecs[vec[0] + 1:]:
            contr_val = contractor(vec[1], v)
            if contr_val != 0:
                curr_contrs[i] = contr_val
        contrs.append(curr_contrs)
        continue
    return vecs, contrs


class _WickExpander:
    """Expander of vectors based on Wick theorem.

    Here, the problem should be specified by a sequence ``vecs``, where each
    entry should be a pair of a unique integral identifier for the vector and
    the actual vector.  The given contractions will be queried with the
    identifier of the first vector for a dictionary of non-vanishing
    contractions with vectors **later** in the sequence.
    """

    def __init__(self, vecs, contrs, phase, contr_all):
        """Initialize the expander."""

        self.vecs = vecs
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
        contrs = self.contrs
        contr_all = self.contr_all

        if pivot == self.n_vecs - 1:
            # Add the current term.
            if not contr_all or all(not i for i in avail):
                expanded.append((
                    amp, [j[1] for i, j in zip(avail, vecs) if i]
                ))
            return

        pivot_idx = vecs[pivot][0]
        pivot_contrs = contrs[pivot_idx]
        if contr_all and len(pivot_contrs) == 0:
            return

        if not contr_all:
            self._add_terms(expanded, amp, avail, pivot + 1)

        avail[pivot_idx] = False
        n_vecs_between = 0
        for vec in vecs[pivot + 1:]:
            vec_idx = vec[0]
            if avail[vec_idx]:
                if vec_idx in pivot_contrs:
                    avail[vec_idx] = False
                    contr_val = pivot_contrs[vec_idx]
                    contr_phase = self.phase ** n_vecs_between
                    self._add_terms(
                        expanded, amp * contr_val * contr_phase,
                        avail, pivot + 1
                    )
                    avail[vec_idx] = True
                n_vecs_between += 1
            continue

        avail[pivot_idx] = True
        return
