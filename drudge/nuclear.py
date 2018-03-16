"""Utilities for nuclear problems."""

import functools
import re

from sympy import (
    Symbol, Function, Sum, symbols, Wild, KroneckerDelta, IndexedBase, Integer
)
from sympy.physics.quantum.cg import CG, Wigner3j, Wigner6j, Wigner9j, cg_simp

from .drudge import Tensor
from .fock import BogoliubovDrudge
from .term import Range
from .utils import sympy_key


class _QNOf(Function):
    """Base class for quantum number access symbolic functions.
    """

    # To be override by subclasses.
    _latex_header = None

    def _latex(self, printer):
        """Form the LaTeX printing.

        The printing will be the header given in the subclasses followed by the
        printing of the orbit wrapped in braces.
        """
        return ''.join([
            self._latex_header, '{', printer.doprint(self.args[0]), '}'
        ])

    def _eval_is_integer(self):
        return True


class JOf(_QNOf):
    """Symbolic access of j quantum number of an orbit."""
    _latex_header = 'j_'


class TildeOf(_QNOf):
    """Symbolic access of the tilde part of an orbit."""
    _latex_header = r'\tilde'


class MOf(_QNOf):
    """Symbolic access of the m quantum number of an orbit."""
    _latex_header = 'm_'


class NOf(_QNOf):
    """Symbolic access of the n quantum number of an orbit."""
    _latex_header = 'n_'


class LOf(_QNOf):
    """Symbolic access of the l quantum number in an orbit."""
    _latex_header = 'l_'


class PiOf(_QNOf):
    """Symbolic access of the parity of an orbit."""
    _latex_header = r'\pi_'


class TOf(_QNOf):
    """Symbolic access to the t quantum number of an orbit."""
    _latex_header = 't_'


_SUFFIXED = re.compile(r'^([a-zA-Z]+)([0-9]+)$')


def _decor_base(symb: Symbol, op, **kwargs):
    """Decorate the base part of the given symbol.

    The given symbol must have a digits-suffixed name, then the given
    operation will be applied to the base part, and recombined with the
    suffix to form the resulted symbol.

    Keyword arguments are all forward to the Symbol constructor.
    """
    m = _SUFFIXED.match(symb.name)
    if not m:
        raise ValueError('Invalid symbol name to parse', symb)

    name, suffix = m.groups()
    return Symbol(op(name) + suffix, **kwargs)


def form_tilde(orig: Symbol):
    """Form the tilde symbol for a given orbit symbol.
    """
    return _decor_base(orig, lambda x: x + 'tilde')


def form_m(orig: Symbol):
    """Form the symbol for m quantum number for a given orbit symbol.
    """
    return _decor_base(orig, lambda _: 'm', integer=True)


class NuclearBogoliubovDrudge(BogoliubovDrudge):
    """Utility drudge for nuclear theories based on Bogoliubov transformation.

    Different from the base :py:class:`BogoliubovDrudge` class, which
    concentrates on the transformation and the commutation rules, here we have
    more utility around the specifics about the nuclear Hamiltonian, especially
    the spherical symmetry.

    .. warning::

        This work is still in progress and highly experimental.  Please at least
        check some of the result before actual usage.

    """

    def __init__(
            self, ctx, coll_j_range=Range('J', 0, Symbol('Jmax') + 1),
            coll_m_range=Range('M'),
            coll_j_dumms=tuple(
                Symbol('J{}'.format(i), integer=True) for i in range(1, 30)
            ),
            coll_m_dumms=tuple(
                Symbol('M{}'.format(i), integer=True) for i in range(1, 30)
            ),
            tilde_range=Range(r'\tilde{Q}'), form_tilde=form_tilde,
            m_range=Range('m'), form_m=form_m, **kwargs
    ):
        """Initialize the drudge object."""
        super().__init__(ctx, **kwargs)

        # Convenient names for quantum number access functions inside drudge
        # scripts.
        self.set_name(
            j_=JOf, tilde_=TildeOf, m_=MOf, n_=NOf, l_=LOf, pi_=PiOf
        )

        self.coll_j_range = coll_j_range
        self.coll_m_range = coll_m_range
        self.coll_j_dumms = coll_j_dumms
        self.coll_m_dumms = coll_m_dumms
        self.set_dumms(coll_j_range, coll_j_dumms)
        self.set_dumms(coll_m_range, coll_m_dumms)

        self.tilde_range = tilde_range
        self.form_tilde = form_tilde
        self.tilde_dumms = tuple(form_tilde(i) for i in self.qp_dumms)
        self.set_dumms(tilde_range, self.tilde_dumms)

        self.m_range = m_range
        self.form_m = form_m
        self.m_dumms = tuple(form_m(i) for i in self.qp_dumms)
        self.set_dumms(m_range, self.m_dumms)

        self.add_resolver_for_dumms()

        # Add utility about CG coefficients and related things.
        self.set_name(
            CG=CG, Wigner3j=Wigner3j, Wigner6j=Wigner6j, Wigner9j=Wigner9j
        )
        self.set_tensor_method('simplify_cg', self.simplify_cg)

        # For angular momentum decoupling.
        #
        # Here is a dictionary giving a callable for all implemented
        # decoupling.
        self._angdec_funcs = {}

        # Cases with total order of 4.
        qp_dumms = self.qp_dumms
        cj = coll_j_dumms[0]
        cm = coll_m_dumms[0]
        jkt1, jkt2, jkt3, jkt4 = [JOf(TildeOf(i)) for i in qp_dumms[:4]]
        mk1, mk2, mk3, mk4 = [MOf(i) for i in qp_dumms[:4]]
        order_4_cases = {
            (0, 4): (_NEG_UNITY ** (jkt1 + jkt2 + cj + cm), {0, 1}),
            (1, 3): (_NEG_UNITY ** (jkt2 - mk2 + cj), {1}),
            (2, 2): (_UNITY, {}),
            (3, 1): (_NEG_UNITY ** (jkt3 - mk3 + cj), {2}),
            (4, 0): (_NEG_UNITY ** (jkt3 + jkt4 + cj + cm), {2, 3}),
        }
        for k, v in order_4_cases.items():
            self._angdec_funcs[k] = functools.partial(
                self._form_angdec_def_4, v[0], v[1]
            )
            continue

        order_2_cases = {
            (0, 2): KroneckerDelta(mk1, -mk2) * _NEG_UNITY ** (jkt2 + mk2),
            (1, 1): KroneckerDelta(mk1, mk2) * _UNITY,
            (2, 0): KroneckerDelta(mk1, -mk2) * _NEG_UNITY ** (jkt1 - mk1)
        }
        for k, v in order_2_cases.items():
            self._angdec_funcs[k] = functools.partial(
                self._form_angdec_def_2, v
            )
            continue
        self.set_tensor_method('do_angdec', self.do_angdec)

    def _form_angdec_def_4(self, phase, neg_ms, base, res_base):
        """Form the angular momentum decoupled form for total order of 4.

        The phase is the over all phase, and the negative m's in the CG
        coefficients can be given in a container, as **zero-based** index.
        """

        ks = self.qp_dumms[:4]
        k1, k2, k3, k4 = ks
        kts = [TildeOf(i) for i in ks]
        kt1, kt2, kt3, kt4 = kts
        jkt1, jkt2, jkt3, jkt4 = [JOf(i) for i in kts]
        # Here the m's already includes the phase.
        mk1, mk2, mk3, mk4 = [
            MOf(ks[i]) * (-1 if i in neg_ms else 1)
            for i in range(4)
        ]
        cj = self.coll_j_dumms[0]
        cm = self.coll_m_dumms[0]

        res = self.define(base[k1, k2, k3, k4], self.sum(
            (cj, self.coll_j_range), (cm, self.coll_m_range[-cj, cj + 1]),
            phase * res_base[cj, kt1, kt2, kt3, kt4] *
            CG(jkt1, mk1, jkt2, mk2, cj, cm) *
            CG(jkt3, mk3, jkt4, mk4, cj, cm)
        ))
        return res

    def _form_angdec_def_2(self, phase, base, res_base):
        """Form the angular momentum decoupled form for total order of 2.

        The phase should include any phase factor except the deltas of pi, J, t
        and the dense tensor indexing.
        """

        ks = self.qp_dumms[:2]
        k1, k2 = ks
        kts = [TildeOf(i) for i in ks]
        kt1, kt2 = kts

        res = self.define(base[k1, k2], self.sum(
            phase * KroneckerDelta(PiOf(kt1), PiOf(kt2))
            * KroneckerDelta(JOf(kt1), JOf(kt2))
            * KroneckerDelta(TOf(kt1), TOf(kt2))
            * res_base[LOf(kt1), JOf(kt1), TOf(kt1), NOf(kt1), NOf(kt2)]
        ))
        return res

    def form_angdec_def(
            self, base: IndexedBase, cr_order: int, an_order: int,
            res_base: IndexedBase = None
    ):
        """Form the tensor definition for angular momentum decoupled form.

        Here for creation and annihilation orders which have been implemented, a
        :py:class:`TensorDef` object will be created as the definition for the
        angular momentum decoupling of the given tensor.  The resulted
        definitions can be used with :py:meth:`do_angdec` method.

        Parameters
        ----------

        base
            The tensor to be rewritten.

        cr_order
            The quasi-particle creation order of the tensor.

        an_order
            The quasi-particle annihilation order of the tensor.

        res_base
            The tensor to be used on the RHS for the tilde tensor.  By default,
            it will have the same name as the LHS.

        """

        res_base = base if res_base is None else res_base
        key = (cr_order, an_order)
        if key not in self._angdec_funcs:
            raise NotImplementedError(
                'Invalid creation/annihilation order to decouple', key
            )
        return self._angdec_funcs[key](base, res_base)

    def do_angdec(self, tensor: Tensor, defs):
        """Expand quasi-particle summation into the tilde and m parts.
        """

        substed = tensor.subst_all(defs)

        # Cache as locals for Spark serialization.
        tilde_range = self.tilde_range
        form_tilde = self.form_tilde
        m_range = self.m_range
        form_m = self.form_m

        def expand(dumm: Symbol):
            """Expand a summation over quasi-particle orbitals."""
            tilde = TildeOf(dumm)
            jtilde = JOf(tilde)
            return [
                (form_tilde(dumm), tilde_range, tilde),
                (form_m(dumm), m_range[-jtilde, jtilde + 1], MOf(dumm))
            ]

        return substed.expand_sums(self.qp_range, expand)

    def simplify_cg(self, tensor: Tensor):
        """Simplify CG coefficients in the expression.

        Here we just concentrate on CG rather than the general case.
        """

        tensor = tensor.map2amps(_canon_cg)

        # Initial simplification of some summations.
        tensor = tensor.simplify_sums(simplify=self._simplify_amp_sum)

        # Deltas could come from some simplification rules.
        tensor = tensor.simplify_deltas()

        # Some summations could become simplifiable after the delta resolution.
        tensor = tensor.simplify_sums()

        return tensor

    @staticmethod
    def _simplify_amp_sum(expr: Sum):
        """Attempt to simplify amplitude sums for nuclear problems.

        Here we specially concentrate on the simplification involving
        Clebosch-Gordan coefficients.  Since this functionality is put into a
        separate tensor function, here we need to invoke it explicitly, since it
        will not be called automatically during the default simplification for
        performance reason.
        """

        # TODO: Add more simplifications here.
        attempts = [
            cg_simp,
            _simpl_varsh_872_4,
            _simpl_varsh_872_5
        ]
        for attempt in attempts:
            res = attempt(expr)
            if res is not None and res != expr:
                return res
            continue

        return None


#
# Angular momentum quantities simplification.
#

def _canon_cg(expr):
    """Pose CG coefficients in the expression into canonical form.
    """
    return expr.replace(CG, _canon_cg_core)


def _canon_cg_core(j1, m1, j2, m2, cj, cm):
    r"""Pose a CG into a canonical form.

    When two of the little :math:`m`s has got negation, we flip the signs of all
    of them.  Then the sort keys of :math:`m_1` and :math:`j_1` will be compared
    with that of :math:`m_2` and :math:`j_2`, which may lead to flipping by
    Varsh 8.4.3 Equation 10.

    """

    phase = _UNITY
    if m1.has(_NEG_UNITY) and m2.has(_NEG_UNITY):
        m1 *= _NEG_UNITY
        m2 *= _NEG_UNITY
        cm *= _NEG_UNITY
        phase *= _NEG_UNITY ** (j1 + j2 - cj)

    if (sympy_key(m1), sympy_key(j1)) > (sympy_key(m2), sympy_key(j2)):
        m1, m2 = m2, m1
        j1, j2 = j2, j1
        phase *= _NEG_UNITY ** (j1 + j2 - cj)

    return CG(j1, m1, j2, m2, cj, cm) * phase


def _simpl_varsh_872_4(expr: Sum):
    """Make CG simplification based on Varsh 8.7.2 Eq (4).

    Compared with the implementation of the same rule in SymPy, here more care
    is taken for better robustness toward different initial arrangements of the
    summations.
    """
    if len(expr.args) != 3:
        return None

    dummies = (expr.args[1][0], expr.args[2][0])
    j1, j2, cj1, cm1, cj2, cm2 = symbols('j1 j2 J1 M1 J2 M2', cls=Wild)

    for m1, m2 in [dummies, reversed(dummies)]:
        match = expr.args[0].match(
            CG(j1, m1, j2, m2, cj1, cm1) * CG(j1, m1, j2, m2, cj2, cm2)
        )
        if not match:
            continue
        return KroneckerDelta(
            match[cj1], match[cj2]
        ) * KroneckerDelta(match[cm1], match[cm2])

    return None


def _simpl_varsh_872_5(expr: Sum):
    """Make CG simplification based on Varsh 8.7.2 Eq (5).
    """
    if len(expr.args) != 3:
        return None

    dummies = (expr.args[1][0], expr.args[2][0])
    j1, j2, m2, j3, m3, cj = symbols('j1 j2 m2 j3 m3 J', cls=Wild)
    for m1, cm in [dummies, reversed(dummies)]:
        match = expr.args[0].match(
            CG(j1, m1, j2, m2, cj, cm) * CG(j1, m1, j3, m3, cj, cm)
        )

        if not match:
            continue

        cjhat = 2 * match[cj] + 1
        jhat2 = 2 * match[j2] + 1

        return (cjhat / jhat2) * KroneckerDelta(
            match[j2], match[j3]
        ) * KroneckerDelta(match[m2], match[m3])

    return None


# Utility constants.

_UNITY = Integer(1)
_NEG_UNITY = Integer(-1)
