"""
Drudge classes for working with operators on Fock spaces.

In this module, several drudge instances are defined for working with creation
and annihilation operators acting on fermion or boson Fock spaces.
"""

import functools
import typing
import warnings

from sympy import KroneckerDelta, IndexedBase, Expr, Symbol, Rational

from ._tceparser import parse_tce_out
from .canon import NEG, IDENT
from .canonpy import Perm
from .drudge import Tensor
from .term import Vec, Range, try_resolve_range, Term
from .utils import sympy_key, ensure_expr, EnumSymbs
from .wick import WickDrudge


#
# General fermion/boson algebra
# -----------------------------
#


class CranChar(EnumSymbs):
    """Transformation characters of creation/annihilation operators.

    It values, which can be accessed as the class attributes ``CR`` and ``AN``
    also forwarded to module scope, should be used as the first index to vectors
    representing creation and annihilation operators.
    """

    _symbs_ = [
        ('CR', r'\dagger'),
        ('AN', '')
    ]


CR = CranChar.CR
AN = CranChar.AN

FERMI = -1
BOSE = 1


class FockDrudge(WickDrudge):
    """Drudge for doing fermion/boson operator algebra on Fock spaces.

    This is the general base class for drudges working on fermion/boson operator
    algebras.  Here general methods are defined for working on these algebraic
    systems, but no problem specific information, like ranges or operator base,
    is defined.

    To customize the details of the commutation rules, properties ``op_parser``
    and ``ancr_contractor`` can be overridden.

    """

    def __init__(self, *args, exch=FERMI, **kwargs):
        """Initialize the drudge.

        Parameters
        ----------

        exch : {1, -1}

            The exchange symmetry for the Fock space.  Constants ``FERMI`` and
            ``BOSE`` can be used.

        """

        super().__init__(*args, **kwargs)
        if exch == FERMI or exch == BOSE:
            self._exch = exch
        else:
            raise ValueError('Invalid exchange', exch, 'expecting plus/minus 1')

        self.set_tensor_method('eval_vev', self.eval_vev)
        self.set_tensor_method('eval_phys_vec', self.eval_phys_vev)

    @property
    def contractor(self):
        """Get the contractor for the algebra.

        The operations are read here on-the-fly so that possibly custemized
        behaviour from the subclasses can be read.
        """

        ancr_contractor = self.ancr_contractor
        op_parser = self.op_parser

        return functools.partial(
            _contr_field_ops, ancr_contractor=ancr_contractor,
            op_parser=op_parser
        )

    @property
    def phase(self):
        """Get the phase for the commutation rules."""
        return self._exch

    @property
    def comparator(self):
        """Get the comparator for the normal ordering operation."""

        op_parser = self.op_parser

        return functools.partial(_compare_field_ops, op_parser=op_parser)

    @property
    def vec_colour(self):
        """Get the vector colour evaluator."""

        op_parser = self.op_parser

        return functools.partial(_get_field_op_colour, op_parser=op_parser)

    OP_PARSER = typing.Callable[
        [Vec, Term], typing.Tuple[typing.Any, CranChar, typing.Sequence[Expr]]
    ]

    @property
    def op_parser(self) -> OP_PARSER:
        """Get the parser for field operators.

        The result should be a callable taking an vector and return a triple of
        operator base, operator character, and the actual indices to the
        operator.  This can be helpful for cases where the interpretation of the
        operators needs to be tweeked.
        """
        return parse_field_op

    ANCR_CONTRACTOR = typing.Callable[
        [typing.Any, typing.Sequence[Expr], typing.Any, typing.Sequence[Expr]],
        Expr
    ]

    @property
    def ancr_contractor(self) -> ANCR_CONTRACTOR:
        """Get the contractor for annihilation and creation operators.

        In this drudge, the contraction between creation/creation,
        annihilation/annihilation, and creation/annihilation operators are
        fixed.  By this property, a callable for contracting annihilation
        operators with a creation operator can be given.  It will be called with
        the base and indices (excluding the character) of the annihilation
        operators and the base and indices of the creation operator.  A simple
        SymPy expression is expected in the result.

        By default, the result will be a simple delta.
        """

        return _contr_ancr_by_delta

    def eval_vev(self, tensor: Tensor, contractor):
        """Evaluate vacuum expectation value.

        The contractor needs to be given as a callable accepting two operators.
        """

        return Tensor(self, self.normal_order(
            tensor.terms, comparator=None, contractor=contractor
        ))

    def eval_phys_vev(self, tensor: Tensor):
        """Evaluate expectation value with respect to the physical vacuum.

        Here the contractor from normal-ordering will be used.
        """

        return Tensor(
            self, self.normal_order(tensor.terms, comparator=None)
        )

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
        or it is assumed to have the same size as the first half.  It can also
        be zero, which gives one chunk of symmetric slots only.
        """

        if n_body < 2:
            raise ValueError(
                'Invalid body count', n_body,
                'expecting a number greater than one'
            )

        n_body2 = n_body if n_body2 is None else n_body2
        n_slots = n_body + n_body2

        transp_acc = NEG if self._exch == FERMI else IDENT
        cycl_accs = [
            NEG if self._exch == FERMI and i % 2 == 0 else IDENT
            for i in [n_body, n_body2]
            ]  # When n_body2 is zero, this value is kinda wrong but not used.

        gens = []

        second_half = list(range(n_body, n_slots))
        gens.append(Perm(
            self._form_cycl(0, n_body) + second_half, cycl_accs[0]
        ))
        gens.append(Perm(
            self._form_transp(0, n_body) + second_half, transp_acc
        ))

        if n_body2 > 1:
            first_half = list(range(0, n_body))
            gens.append(Perm(
                first_half + self._form_cycl(n_body, n_slots), cycl_accs[1]
            ))
            gens.append(Perm(
                first_half + self._form_transp(n_body, n_slots), transp_acc
            ))

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

    def _latex_vec(self, vec):
        """Get the LaTeX form of field operators.
        """

        head = r'{}^{{{}}}'.format(vec.label, self._latex_sympy(vec.indices[0]))
        indices = ', '.join(self._latex_sympy(i) for i in vec.indices[1:])
        return r'{}_{{{}}}'.format(head, indices)

    _latex_vec_mul = ' '


def parse_field_op(op: Vec, _: Term):
    """Get the operator label, character and actual indices.

    ValueError will be raised if the given operator does not satisfy the format
    for field operators.
    """

    indices = op.indices
    if len(indices) < 1 or (indices[0] != CR and indices[0] != AN):
        raise ValueError('Invalid field operator', op,
                         'expecting operator character')

    return op.label, indices[0], indices[1:]


def _compare_field_ops(
        op1: Vec, op2: Vec, term: Term,
        op_parser: FockDrudge.OP_PARSER
):
    """Compare the given field operators.

    Here we try to emulate physicists' convention as much as possible.  The
    annihilation operators are ordered in reversed direction.
    """

    label1, char1, indices1 = op_parser(op1, term)
    label2, char2, indices2 = op_parser(op2, term)

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


def _contr_field_ops(op1: Vec, op2: Vec, term: Term,
                     ancr_contractor: FockDrudge.ANCR_CONTRACTOR,
                     op_parser: FockDrudge.OP_PARSER):
    """Contract two field operators.

    Here we work by the fermion-boson commutation rules.  The contractor is only
    going to be called for annihilation creation pairs, with all others implied
    by the algebra.

    """

    label1, char1, indices1 = op_parser(op1, term)
    label2, char2, indices2 = op_parser(op2, term)

    if char1 == char2 or char1 == CR:
        return 0

    return ancr_contractor(label1, indices1, label2, indices2)


def _contr_ancr_by_delta(label1, indices1, label2, indices2):
    """Contract an annihilation and a creation operator by delta."""

    # For the delta contraction, some additional checking is needed for it to
    # make sense.

    err_header = 'Invalid field operators to contract by delta'

    # When the operators are on different base, it is likely that delta is not
    # what is intended.

    if label1 != label2:
        raise ValueError(err_header, (label1, label2),
                         'expecting the same base')

    if len(indices1) != len(indices2):
        raise ValueError(err_header, (indices1, indices2),
                         'expecting same number of indices')

    res = 1
    for i, j in zip(indices1, indices2):
        # TODO: Maybe support continuous indices here.
        res *= KroneckerDelta(i, j)
        continue

    return res


def _get_field_op_colour(idx, vec, term, op_parser: FockDrudge.OP_PARSER):
    """Get the colour of field operators.

    Here the annihilation part is specially treated for better compliance with
    conventions in physics.
    """

    _, char, _ = op_parser(vec, term)
    return char, idx if char == CR else -idx


#
# Detailed problems
# -----------------
#


def conserve_spin(*spin_symbs):
    """Get a callback giving true only if the given spin values are conserved.

    Here by conserving the spin, we mean the very strict sense that the values
    of the first half of the symbols in the given dictionary exactly matches the
    corresponding values in the second half.
    """

    n_symbs = len(spin_symbs)
    if n_symbs % 2 == 1:
        raise ValueError('Invalid spin symbols', spin_symbs,
                         'expecting a even number of them')
    n_particles = n_symbs // 2

    outs = spin_symbs[0:n_particles]
    ins = spin_symbs[n_particles:n_symbs]

    def test_conserve(symbs_dict):
        """Test if the spin values from the dictionary is conserved."""
        return all(
            symbs_dict[i] == symbs_dict[j]
            for i, j in zip(ins, outs)
        )

    return test_conserve


class GenMBDrudge(FockDrudge):
    """Drudge for general many-body problems.

    In a general many-body problem, a state for the particle is given by a
    symbolic **orbital** quantum numbers for the external degrees of freedom and
    optionally a concrete **spin** quantum numbers for the internal states of
    the particles.  Normally, there is just one orbital quantum number and one
    or no spin quantum number.

    In this model, a default Hamiltonian of the model is constructed from a
    one-body and two-body interaction, both of them are assumed to be spin
    conserving.

    """

    def __init__(self, *args, exch=FERMI, op_label='c',
                 orb=((Range('L'), 'abcdefghijklmnopq'),), spin=(),
                 one_body=IndexedBase('t'), two_body=IndexedBase('u'),
                 dbbar=False, **kwargs):
        """Initialize the drudge object.

        TODO: Add details documentation here.
        """

        super().__init__(*args, exch=exch, **kwargs)

        #
        # Create the field operator
        #

        op = Vec(op_label)
        cr = op[CR]
        an = op[AN]
        self.op = op
        self.cr = cr
        self.an = an
        # Register the name of the operator later to avoid being shallowed by
        # dummies.

        #
        # Hamiltonian creation
        #
        # Other aspects of the model will also be set during this stage.
        #

        orb_ranges = []
        for range_, dumms in orb:
            self.set_dumms(range_, dumms)
            orb_ranges.append(range_)
            continue
        self.orb_ranges = orb_ranges
        self.add_resolver_for_dumms()

        # Register core field operator name.
        self.set_name(an, str(op) + '_')
        self.set_name(cr, str(op) + '_dag')

        spin_vals = []
        for i in spin:
            spin_vals.append(ensure_expr(i))
        has_spin = len(spin_vals) > 0
        if len(spin_vals) == 1:
            warnings.warn(
                'Just one spin value is given: '
                'consider dropping it for better performance'
            )
        self.spin_vals = spin_vals

        # These dummies are used temporarily and will soon be reset.  They are
        # here, rather than the given dummies directly, because we might need to
        # dummy for multiple orbital ranges.
        #
        # They are created as tuple so that they can be easily used for
        # indexing.

        orb_dumms = tuple(
            Symbol('internalOrbitPlaceholder{}'.format(i))
            for i in range(4)
        )
        spin_dumms = tuple(
            Symbol('internalSpinPlaceholder{}'.format(i))
            for i in range(4)
        )

        orb_sums = [(i, orb_ranges) for i in orb_dumms]
        spin_sums = [(i, spin_vals) for i in spin_dumms]

        # The indices to get the operators in the hamiltonian.
        indices = [
            (i, j) if has_spin else i
            for i, j in zip(orb_dumms, spin_dumms)
            ]

        # Actual Hamiltonian building.

        self.one_body = one_body
        self.set_name(one_body)  # No symmetry for it.

        one_body_sums = orb_sums[:2]
        if has_spin:
            one_body_sums.extend(spin_sums[:2])

        one_body_ham = self.sum(
            *one_body_sums,
            one_body[orb_dumms[:2]] * cr[indices[0]] * an[indices[1]],
            predicate=conserve_spin(*spin_dumms[:2]) if has_spin else None
        )

        if dbbar:
            two_body_coeff = Rational(1, 4)
            self.set_dbbar_base(two_body, 2)
        else:
            two_body_coeff = Rational(1, 2)
            self.set_n_body_base(two_body, 2)
        self.two_body = two_body

        two_body_sums = orb_sums
        if has_spin:
            two_body_sums.extend(spin_sums)

        two_body_ham = self.sum(
            *two_body_sums,
            two_body_coeff * two_body[orb_dumms] *
            cr[indices[0]] * cr[indices[1]] * an[indices[3]] * an[indices[2]],
            predicate=conserve_spin(*spin_dumms) if has_spin else None
        )

        # We need to at lease remove the internal symbols.
        orig_ham = (one_body_ham + two_body_ham).reset_dumms()
        self.orig_ham = orig_ham

        simpled_ham = orig_ham.simplify()
        self.ham = simpled_ham


class PartHoleDrudge(GenMBDrudge):
    """Drudge for the particle-hole problems.

    This model contains different forms of the Hamiltonian.

    orig_ham

        The original form of the Hamiltonian, written in terms of bare one-body
        and two-body interaction tensors without normal-ordering with respect to
        the Fermion vacuum.

    full_ham

        The full form of the Hamiltonian in terms of the bare interaction
        tensors, normal-ordered with respect to the Fermi vacuum.

    ham_energy

        The zero energy inside the full Hamiltonian.

    one_body_ham

        The one-body part of the full Hamiltonian, written in terms of the bare
        interaction tensors.

    ham

        The most frequently used form of the Hamiltonian, written in terms of
        Fock matrix and the two-body interaction tensor.


    """

    DEFAULT_PART_DUMMS = tuple(Symbol(i) for i in 'abcdefgh') + tuple(
        Symbol('a{}'.format(i)) for i in range(50)
    )
    DEFAULT_HOLE_DUMMS = tuple(Symbol(i) for i in 'ijklmnpq') + tuple(
        Symbol('i{}'.format(i)) for i in range(50)
    )

    def __init__(self, *args, op_label='c',
                 part_orb=(Range('V'), DEFAULT_PART_DUMMS),
                 hole_orb=(Range('O'), DEFAULT_HOLE_DUMMS),
                 spin=(),
                 one_body=IndexedBase('t'), two_body=IndexedBase('u'),
                 fock=IndexedBase('f'),
                 dbbar=True, **kwargs):
        """Initialize the particle-hole drudge."""

        self.part_range = part_orb[0]
        self.hole_range = hole_orb[0]

        super().__init__(*args, exch=FERMI, op_label=op_label,
                         orb=(part_orb, hole_orb), spin=spin,
                         one_body=one_body, two_body=two_body, dbbar=dbbar,
                         **kwargs)

        full_ham = self.ham
        full_ham.cache()
        self.full_ham = full_ham

        self.ham_energy = full_ham.filter(lambda term: term.is_scalar)

        self.one_body_ham = full_ham.filter(
            lambda term: len(term.vecs) == 2
        )
        two_body_ham = full_ham.filter(lambda term: len(term.vecs) == 4)

        # We need to rewrite the one-body part in terms of Fock matrices.
        self.fock = fock
        one_body_terms = []
        for i in self.one_body_ham.local_terms:
            if i.amp.has(one_body):
                one_body_terms.append(i.subst({one_body: fock}))
            continue
        rewritten_one_body_ham = self.create_tensor(one_body_terms)

        ham = rewritten_one_body_ham + two_body_ham
        ham.cache()
        self.ham = ham

        self.set_tensor_method('eval_fermi_vev', self.eval_fermi_vev)

    @property
    def op_parser(self):
        """Get the special operator parser for particle-hole problems.

        Here when the first index to the operator is resolved to be a hole
        state, the creation/annihilation character of the operator will be
        flipped.
        """

        resolvers = self.resolvers
        hole_range = self.hole_range

        def parse_parthole_ops(op: Vec, term: Term):
            """Parse the operator for particle/hole field operator."""
            label, char, indices = parse_field_op(op, term)
            orb_range = try_resolve_range(
                indices[0], dict(term.sums), resolvers.value
            )
            if orb_range is None:
                raise ValueError('Invalid orbit value', indices[0],
                                 'expecting particle or hole')
            if orb_range == hole_range:
                char = AN if char == CR else CR
            return label, char, indices

        return parse_parthole_ops

    def eval_fermi_vev(self, tensor: Tensor):
        """Evaluate expectation value with respect to Fermi vacuum.

        This is just an alias to the actual `eval_phys_vev` method to avoid
        confusion about the terminology in particle-hole problems.
        """
        return self.eval_phys_vev(tensor)

    def parse_tce(self, tce_out: str,
                  cc_bases: typing.Mapping[int, IndexedBase]):
        """Parse TCE output into a tensor.

        The CC amplitude bases should be given as a dictionary mapping from the
        excitation order to the actual base.
        """

        def range_cb(label):
            """The range call-back."""
            return self.part_range if label[0] == 'p' else self.hole_range

        def base_cb(name, indices):
            """Get the indexed base for a name in TCE output."""
            if name == 'f':
                return self.fock
            elif name == 'v':
                return self.two_body
            elif name == 't':
                return cc_bases[len(indices) // 2]
            else:
                raise ValueError('Invalid base', name, 'in TCE output.')

        terms, free_vars = parse_tce_out(tce_out, range_cb, base_cb)

        # Here we assume that the symbols from directly conversion from TCE
        # output will not conflict with the canonical dummies.
        substs = {}
        for range_, symbs in free_vars.items():
            for i, j in zip(
                    sorted(symbs, key=lambda x: int(x.name[1:])),
                    self.dumms.value[range_]
            ):
                substs[i] = j
                continue
            continue

        return self.create_tensor([i.subst(substs) for i in terms])
