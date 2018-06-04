"""Canonicalization of tensorial factors.

This module supports the canonicalization of tensorial quantities,
by delegating the actual work to the core canonpy module.

"""

import itertools
import typing
import warnings

from sympy import conjugate, Symbol

from .canonpy import canon_eldag, Group, Perm
from .utils import sympy_key

#
# Preparation
# -----------
#

# Actions.
IDENT = 0
NEG = 1
CONJ = 2


class Eldag:
    """A shallow container for information about an Eldag.

    This class is designed more toward the cases where the Eldag is built
    one node after another.

    """

    def __init__(self):
        """Initialize the Eldag."""

        self.edges = []
        self.ia = [0]
        self.symms = []
        self.colours = []

    def add_node(self, edges: typing.Iterable[int], symm, colour) -> int:
        """Add a node.

        The index of the given node will be returned.
        """
        self.edges.extend(edges)
        self.ia.append(len(self.edges))
        self.symms.append(symm)
        self.colours.append(colour)

        return len(self.symms) - 1

    @property
    def int_colour(self):
        """Get the integral form of the current node colours."""

        int_colour = [None for _ in self.colours]

        group_res = enumerate(itertools.groupby(
            sorted((v, i) for i, v in enumerate(self.colours)),
            lambda x: x[0]
        ))

        for i, v in group_res:
            _, g = v
            for _, idx in g:
                int_colour[idx] = i
                continue
            continue

        return int_colour

    def canon(self):
        """Canonicalize the Eldag.

        The canonicalization result from canonpy is directly returned.
        """

        return canon_eldag(self.edges, self.ia, self.symms, self.int_colour)


# Node labels.

_SUM = 0
_EXPR = 1
_FACTOR = 2


#
# Driver
# ------
#


def canon_factors(sums, factors, symms):
    """Canonicalize the factors.

    The factors should be a iterable of factor/colour pairs, where the factor
    can be anything with the ``base`` and ``indices`` methods implemented, or
    ``func`` and ``args`` implemented.  It is designed primarily to work with
    SymPy Indexed quantities and vectors, or general SymPy function expressions.
    The colour should be totally ordered, and they only need to order within
    factors correctly.

    The new canonicalized list of summations and canonicalized factors are
    going to be returned.  Also returned is a coefficient that need to be
    multiplied to the amplitude, which is from anti-commutative quantities.

    The symmetries should be given as a mapping from the *base* of the factors
    to the actual symmetries.  A given valence can also be given by a tuple with
    the actual base.

    """

    from .term import Vec

    # They need to be looped over multiple times.
    sums = list(sums)
    factors = list(factors)

    # TODO: make handling of empty eldags more elegant.
    if len(factors) == 0 and len(sums) == 0:
        return sums, factors, 1

    eldag, factor_idxes = _build_eldag(sums, factors, symms)
    node_order, perms = eldag.canon()

    # Sums are guaranteed to be in the initial segment of the nodes, but they
    # might not be at the beginning any more after the canonicalization.
    sums_res = [sums[i] for i in node_order if
                eldag.colours[i][0] == _SUM]

    coeff = 1
    factors_res = []

    for i, v in enumerate(factors):

        factor = v[0]
        if hasattr(factor, 'indices'):
            indices = factor.indices
            is_indexed = True
        else:
            indices = factor.args
            is_indexed = False
        valency = len(indices)
        perm = perms[factor_idxes[i]]
        if_vector = isinstance(factor, Vec)

        if valency < 2 or perm is None:
            factor_res = factor
        else:
            new_indices = tuple(indices[perm[i]] for i in range(valency))
            if is_indexed:
                factor_res = factor.base[new_indices]
            else:
                factor_res = factor.func(new_indices)

            acc = perm.acc
            if acc & NEG:
                if if_vector:
                    coeff *= -1
                else:
                    factor_res = -factor_res
            if acc & CONJ:
                # TODO: Allow vectors to have their own dagger form, maybe.
                if if_vector:
                    raise ValueError(
                        'Vector', factor, 'cannot have conjugation symmetry'
                    )
                factor_res = conjugate(factor_res)

        factors_res.append(factor_res)
        continue

    return sums_res, factors_res, coeff


#
# Internals
# ---------
#


def _build_eldag(sums, factors, symms):
    """Build the eldag for the factors.

    The summations will be put as the first nodes.  Then each factor is treated
    one by one, with its indices coming before itself.

    """

    eldag = Eldag()
    factor_idxes = []

    # Simple treatment of edges among sums: a sum has an edge to another sum if
    # and only if the dummy of the other sum appears in its bounds.
    #
    # In this way, for most problem with no relation among the summations,
    # basically no overhead is introduced.  And it is generally sufficient for
    # simple relationships among the summations.
    #
    # TODO: Better treatment of dummies in summation bounds.

    idx_of_dummies = {v[0]: i for i, v in enumerate(sums)}

    # No need to touch edges for sums.
    for _, i in sums:
        edges = []
        free_var_keys = []  # As part of the colour.
        with_dummy = 0
        if i.bounded:
            bounded = 1
            symbs = i.lower.atoms(Symbol) | i.upper.atoms(Symbol)
            for j in symbs:
                if j in idx_of_dummies:
                    with_dummy = 1
                    edges.append(idx_of_dummies[j])
                else:
                    free_var_keys.append(sympy_key(j))
        else:
            bounded = 0
            edges = []

        free_var_keys.sort()
        # Unbounded comes before bounded, those without dummy involvement comes
        # before those with.
        eldag.add_node(edges, None, (
            _SUM, i.label, bounded, with_dummy, free_var_keys
        ))
        continue

    # Real work, factors.
    #
    # From symbol to node.
    dumms = {v[0]: i for i, v in enumerate(sums)}

    for factor, colour in factors:
        if hasattr(factor, 'base'):
            base = factor.base
            indices = factor.indices
        else:
            base = factor.func
            indices = factor.args
        n_indices = len(indices)

        if n_indices < 2:
            factor_symms = None
        else:
            prim_keys = [base]
            if hasattr(base, 'label'):
                prim_keys.append(base.label)
            keys = itertools.chain(
                ((i, n_indices) for i in prim_keys),
                prim_keys
            )
            for i in keys:
                if i in symms:
                    factor_symms = symms[i]
                    break
                else:
                    continue
            else:
                factor_symms = None

        index_nodes = _proc_indices(indices, dumms, eldag)
        idx = eldag.add_node(
            index_nodes, factor_symms, (_FACTOR, colour)
        )

        factor_idxes.append(idx)
        continue

    return eldag, factor_idxes


class _Placeholders(dict):
    """The dictionary of placeholders for dummies."""

    def __missing__(self, key):
        """Add the placeholder for the given dummy."""
        return Symbol('internalDummyPlaceholder{}'.format(key))


_placeholders = _Placeholders()


def _proc_indices(indices, dumms, eldag):
    """Process the indices to a given factor.

    The symmetry of the expressions at the indices with respect to the dummies
    are fully treated.
    """
    nodes = []

    for expr in indices:

        involved = {}  # Sum node index -> actual dummy.
        for i in expr.atoms(Symbol):
            if i in dumms:
                involved[dumms[i]] = i
            continue

        sum_nodes = list(involved.keys())

        curr_form = None
        curr_order = None
        curr_edges = None
        curr_symms = []

        if len(sum_nodes) > 2:
            warnings.warn(
                "Index expression", expr,
                "contains too many summed dummies, something might be wrong"
            )

        for edges in itertools.permutations(sum_nodes):
            substs = {
                involved[v]: _placeholders[i]
                for i, v in enumerate(edges)
            }
            form = expr.xreplace(substs)

            order = sympy_key(form)
            if curr_form is None or order < curr_order:
                curr_form = form
                curr_order = order
                curr_edges = edges
                curr_symms = []
            elif form == curr_form:
                curr_symms.append(_find_perm(curr_edges, edges))
            continue

        # Now the order of the edges are determined.

        idx = eldag.add_node(
            curr_edges, Group(curr_symms) if len(curr_symms) > 0 else None,
            (_EXPR, curr_order)
        )

        nodes.append(idx)
        continue

    return nodes


def _find_perm(orig, dest):
    """Find the permutation bringing the original sequence to the target.

    Internal function, no checking.
    """

    idxes = {v: i for i, v in enumerate(orig)}
    return Perm(idxes[i] for i in dest)
