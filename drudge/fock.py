"""
Drudge classes for working with operators on Fock spaces.

In this module, several drudge instances are defined for working with creation
and annihilation operators acting on fermion or boson Fock spaces.
"""

import functools

from sympy import Integer, KroneckerDelta, IndexedBase

from .canon import NEG, IDENT
from .canonpy import Perm
from .drudge import Tensor
from .term import Vec
from .utils import sympy_key
from .wick import WickDrudge, wick_expand


#
# General fermion/boson algebra
# -----------------------------
#


class _OpChar(Integer):
    """Transformation characters of vectors.

    The purpose of this class is mostly twofold.  First, better string
    representation of its values can be possible.  Second, simple comparison can
    be used to put creation operators ahead of annihilation operators.

    This class is kept private, while its two singleton constant instances are
    exposed.
    """

    CR = -1
    AN = 1

    def __new__(cls, val):
        """Create an operator character."""
        return super().__new__(cls, val)

    def __str__(self):
        """Get the string representation of the character."""
        if self == self.CR:
            return 'CR'
        elif self == self.AN:
            return 'AN'
        else:
            assert False

    __repr__ = __str__


CR = _OpChar(_OpChar.CR)
AN = _OpChar(_OpChar.AN)

FERMI = -1
BOSE = 1


class FockDrudge(WickDrudge):
    """Drudge for doing fermion/boson operator algebra on Fock spaces.

    This is the general base class for drudges working on fermion/boson operator
    algebras.  Here general methods are defined for working on these algebraic
    systems, but no problem specific information, like ranges or operator base,
    is defined.

    """

    def __init__(self, ctx, exch, contractor=None):
        """Initialize the drudge.

        Parameters
        ----------

        exch : {1, -1}

            The exchange symmetry for the Fock space.  Constants ``FERMI`` and
            ``BOSE`` can be used.

        contractor : Callable

            A callable going to be called with an annihilation operator and a
            creation operator to get their contraction.  By default, delta
            values are assumed.

        """

        super().__init__(ctx)
        if exch == FERMI or exch == BOSE:
            self._exch = exch
        else:
            raise ValueError('Invalid exchange', exch, 'expecting plus/minus 1')

        self._contractor = functools.partial(
            _contr_field_ops, contractor=contractor
        )

    @property
    def contractor(self):
        """Get the contractor for the algebra."""
        return self._contractor

    @property
    def phase(self):
        """Get the phase for the commutation rules."""
        return self._exch

    @property
    def comparator(self):
        """Get the comparator for the normal ordering operation."""
        return _compare_field_ops

    @property
    def vec_colour(self):
        """Get the vector colour evaluator."""
        return _get_field_op_colour

    def eval_vev(self, tensor: Tensor, contractor):
        """Evaluate vacuum expectation value.

        The contractor needs to be given as a callable accepting two operators.
        """

        term_op = functools.partial(
            wick_expand, comparator=None,
            contractor=contractor, phase=self.phase
        )

        return tensor.apply(lambda terms: terms.flatMap(term_op))

    def set_n_body_base(self, base: IndexedBase, n_body: int):
        """Set an indexed base as an n-body interaction.

        The symmetry of an n-body interaction has full permutation symmetry
        among the corresponding slots in the first and second half.

        When the body count if less than two, no symmetry is added.

        """

        # No symmtry going to be added for less than two body.
        if n_body < 2:
            return

        begin1 = 0
        end1 = n_body
        begin2 = end1
        end2 = 2 * n_body

        cycl = Perm(
            self._form_cycl(begin1, end1) + self._form_cycl(begin2, end2)
        )
        transp = Perm(
            self._form_transp(begin1, end1) + self._form_transp(begin2, end2)
        )

        self.set_symm(base, cycl, transp)

        return

    def set_dbbar_base(self, base: IndexedBase, n_body: int, n_body2=None):
        """Set an indexed base as a double-bar interaction.

        A double barred interaction has full permutation symmetry among its
        first half of slots and its second half individually.  For fermion
        field, the permutation is assumed to be anti-commutative.

        The size of the second half can be given by another optional argument,
        or it is assumed to have the same size as the first half.
        """

        n_body2 = n_body if n_body2 is None else n_body2

        gens = []
        begin = 0
        for i in [n_body, n_body2]:
            end = begin + i
            if i > 1:
                cycl_accs = NEG if self._exch == FERMI and i % 2 == 0 else IDENT
                transp_acc = NEG if self._exch == FERMI else IDENT
                gens.append(Perm(
                    self._form_cycl(begin, end), cycl_accs
                ))
                gens.append(Perm(
                    self._form_transp(begin, end), transp_acc
                ))
            begin = end

        self.set_symm(base, gens)

        return

    @staticmethod
    def _form_cycl(begin, end):
        """Form the pre-image for a cyclic permutation over the given range."""
        before_end = end - 1
        res = [before_end]
        res.extend(range(begin, before_end))
        return res

    @staticmethod
    def _form_transp(begin, end):
        """Form a pre-image array with the first two points transposed."""
        res = list(range(begin, end))
        res[0], res[1] = res[1], res[0]
        return res


def parse_field_op(op: Vec):
    """Get the operator label, character and actual indices.

    ValueError will be raised if the given operator does not satisfy the format
    for field operators.
    """

    indices = op.indices
    if len(indices) < 1 or (indices[0] != CR and indices[0] != AN):
        raise ValueError('Invalid field operator', op,
                         'expecting operator character')

    return op.label, indices[0], indices[1:]


def _compare_field_ops(op1: Vec, op2: Vec):
    """Compare the given field operators.

    Here we try to emulate physicists' convention as much as possible.  The
    annihilation operators are ordered in reversed direction.
    """

    label1, char1, indices1 = parse_field_op(op1)
    label2, char2, indices2 = parse_field_op(op2)

    if char1 == CR and char2 == AN:
        return True
    elif char1 == AN and char2 == CR:
        return False

    key1 = (label1, [sympy_key(i) for i in indices1])
    key2 = (label2, [sympy_key(i) for i in indices2])

    # Equal key are always true for stable insert sort.
    if char1 == CR:
        return key1 <= key2
    else:
        return key1 >= key2


def _contr_field_ops(op1: Vec, op2: Vec, contractor=None):
    """Contract two field operators.

    Here we work by the fermion-boson commutation rules.  The contractor is only
    going to be called for annihilation creation pairs, with all others implied
    by the algebra.

    """

    label1, char1, indices1 = parse_field_op(op1)
    label2, char2, indices2 = parse_field_op(op2)

    if char1 == char2 or char1 == CR:
        return 0

    if contractor is not None:
        return contractor(op1, op2)

    # Else, internal delta contraction is attempted.  For the delta contraction,
    # some additional checking is needed for it to make sense.

    err_header = 'Invalid field operators to contract by delta'

    # When the operators are on different base, it is likely that delta is not
    # what is intended.

    if label1 != label2:
        raise ValueError(err_header, op1, op2, 'expecting the same base')

    if len(indices1) != len(indices2):
        raise ValueError(err_header, op1, op2,
                         'expecting same number of indices')

    res = 1
    for i, j in zip(indices1, indices2):
        # TODO: Maybe support continuous indices here.
        res *= KroneckerDelta(i, j)
        continue

    return res


def _get_field_op_colour(idx, vec):
    """Get the colour of field operators.

    Here the annihilation part is specially treated for better compliance with
    conventions in physics.
    """

    _, char, _ = parse_field_op(vec)
    return char, idx if char == CR else -idx

#
# Detailed problems
# -----------------
#
