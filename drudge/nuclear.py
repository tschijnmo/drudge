"""Utilities for nuclear problems."""

import re

from sympy import (
    Symbol, Function, Sum, symbols, Wild, KroneckerDelta, IndexedBase, Integer,
    sqrt
)
from sympy.physics.quantum.cg import CG, Wigner3j, Wigner6j, Wigner9j

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
    more utilities around the specifics about the nuclear Hamiltonian,
    especially the spherical symmetry.  Notably, the single particle states are
    assumed to be labels by quantum numbers according to the Baranger scheme
    [Suhonen]_, where each single-particle state can be labeled by the orbit
    angular momentum :math:`l`, the total angular momentum :math:`j`, the
    :math:`z`-component of the total angular momentum :math:`m`, and the
    principal quantum number :math:`n`.  The bundle of all the quantum numbers
    can be given as a single symbolic quantum number over a symbolic range, as
    in the base class, while the bundle of quantum numbers other than :math:`m`
    can also be given by tilde symbols over the corresponding tilde range.

    With this Baranger scheme, some utilities for performing angular momentum
    coupling for interaction tensors are also provided.

    .. [Suhonen] J Suhonen, From Nucleons to Nucleus, Springer 2007.

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

        self._cg_sum_simplifiers = {
            # TODO: Add more simplifications here.
            2: [_simpl_varsh_872_4, _simpl_varsh_872_5],
            5: [_simpl_varsh_911_8]
        }
        self.set_tensor_method('simplify_cg', self.simplify_cg)

        # For angular momentum coupling.
        self.set_tensor_method('do_amc', self.do_amc)

    #
    # Angular momentum coupling utilities
    #

    def form_amc_def(
            self, base: IndexedBase, cr_order: int, an_order: int,
            res_base: IndexedBase = None
    ):
        """Form the tensor definition for angular momentum coupled form.

        Here for creation and annihilation orders which have been implemented, a
        :py:class:`TensorDef` object will be created for the definition of the
        original tensor in terms of the angular momentum coupled form of the
        given tensor.

        The resulted definitions are all written in terms of component accessor
        functions on the original bundled symbolic quantum numbers.  When atomic
        symbols are desired for the :math:`m` component, with the rest grouped
        into a tilde symbol, the :py:meth:`do_amc` method can be used together
        with the current method.

        Currently, only coupling for 1- and 2-body tensors are supported.

        TODO: Add description of the form of coupling and make it more tunable.

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
        total_order = cr_order + an_order

        if total_order == 2:
            return self._form_amc_def_1(base, cr_order, an_order, res_base)
        elif total_order == 4:
            return self._form_amc_def_2(base, cr_order, an_order, res_base)
        else:
            raise NotImplementedError(
                'AMC has not been implemented for total order', total_order
            )

    def _form_amc_def_1(self, base, cr_order, an_order, res_base):
        """Form AMC for 1-body tensors.
        """
        assert cr_order + an_order == 2

        k1, k2 = self.qp_dumms[:2]
        mk1, mk2 = MOf(k1), MOf(k2)

        phase = _UNITY
        if cr_order == 2:
            phase = _NEG_UNITY ** (JOf(k2) + mk2),
            mk2 = -mk2
        elif cr_order == 1:
            pass
        elif cr_order == 0:
            phase = _NEG_UNITY ** (JOf(k1) - mk1)
            mk1 = -mk1
        else:
            assert 0

        res = self.define(base[k1, k2], self.sum(
            phase * KroneckerDelta(PiOf(k1), PiOf(k2))
            * KroneckerDelta(JOf(k1), JOf(k2))
            * KroneckerDelta(TOf(k1), TOf(k2))
            * KroneckerDelta(mk1, mk2)
            * res_base[LOf(k1), JOf(k1), TOf(k1), NOf(k1), NOf(k2)]
        ))
        return res

    def _form_amc_def_2(self, base, cr_order, an_order, res_base):
        """Form AMC for 2-body tensors.
        """
        assert cr_order + an_order == 4

        ks = self.qp_dumms[:4]
        k1, k2, k3, k4 = ks
        mk1, mk2, mk3, mk4 = [MOf(i) for i in ks]
        jk1, jk2, jk3, jk4 = [JOf(i) for i in ks]
        cj = self.coll_j_dumms[0]
        cm = self.coll_m_dumms[0]

        phase = _UNITY
        if cr_order == 0:
            phase = _NEG_UNITY ** (jk1 + jk2 + cj + cm)
            mk1 = -mk1
            mk2 = -mk2
        elif cr_order == 1:
            phase = _NEG_UNITY ** (jk2 - mk2 + cj)
            mk2 = -mk2
        elif cr_order == 2:
            pass
        elif cr_order == 3:
            phase = _NEG_UNITY ** (jk3 - mk3 + cj)
            mk3 = -mk3
        elif cr_order == 4:
            phase = _NEG_UNITY ** (jk3 + jk4 + cj + cm)
            mk3 = -mk3
            mk4 = -mk4

        res = self.define(base[k1, k2, k3, k4], self.sum(
            (cj, self.coll_j_range), (cm, self.coll_m_range[-cj, cj + 1]),
            phase
            * res_base[cj, TildeOf(k1), TildeOf(k2), TildeOf(k3), TildeOf(k4)]
            * CG(jk1, mk1, jk2, mk2, cj, cm)
            * CG(jk3, mk3, jk4, mk4, cj, cm)
        ))
        return res

    def do_amc(self, tensor: Tensor, defs, exts=None):
        """Expand quasi-particle summation into the tilde and m parts.

        This is a small convenience utility for angular momentum coupling of
        tensors inside :py:class:`Tensor` objects.  The given definitions will
        all be substituted in, and the original symbols for bundled quantum
        numbers will be separated into the tilde symbols and the atomic
        :math:`m` symbols.

        Parameters
        ----------

        tensor
            The tensor to be substituted.

        defs
            The definitions to be substituted in, generally from
            :py:meth:`form_amc_def` method.

        exts
            External symbols that is also going to be decomposed.

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

        return substed.expand_sums(
            self.qp_range, expand, exts=exts, conv_accs=[NOf, LOf, JOf, TOf]
        )

    def simplify_cg(self, tensor: Tensor):
        """Simplify CG coefficients in the expression.

        Here we specially concentrate on the simplification involving
        Clebosch-Gordan coefficients.  Since this functionality is put into a
        separate tensor function, here we need to invoke it explicitly, since it
        will not be called automatically during the default simplification for
        performance reason.
        """

        tensor = tensor.map2amps(_canon_cg)

        # Initial simplification of some summations.
        tensor = tensor.simplify_sums(simplifiers=self._cg_sum_simplifiers)

        # Deltas could come from some simplification rules.
        tensor = tensor.simplify_deltas()

        # Some summations could become simplifiable after the delta resolution.
        tensor = tensor.simplify_sums()

        return tensor


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


def _simpl_varsh_911_8(expr: Sum):
    """Make CG simplification based on Varsh 9.1.1 Eq (8).
    """
    if len(expr.args) != 6:
        return None

    j, m, j12, m12, j2, m2 = symbols('j m j12 m12 j2 m2', cls=Wild)
    j1, m1 = symbols('j1 m1', cls=Wild)
    j_prm, m_prm, j22, m22 = symbols('jprm mprm j22 m22', cls=Wild)
    j23, m23, j3, m3 = symbols('j23 m23 j3 m3', cls=Wild)

    match = expr.args[0].match(
        CG(j12, m12, j3, m3, j, m) * CG(j1, m1, j2, m2, j12, m12) *
        CG(j1, m1, j23, m23, j_prm, m_prm) * CG(j2, m2, j3, m3, j23, m23)
    )

    if not match or sorted((match[i] for i in (
            m1, m2, m3, m12, m23
    )), key=sympy_key) != sorted((i[0] for i in expr.args[1:]), key=sympy_key):
        return None

    jhat12 = sqrt(2 * match[j12] + 1)
    jhat23 = sqrt(2 * match[j23] + 1)

    phase = _NEG_UNITY ** (match[j1] + match[j2] + match[j3] + match[j])

    return jhat12 * jhat23 * phase * KroneckerDelta(
        match[j], match[j_prm]
    ) * KroneckerDelta(match[m], match[m_prm]) * Wigner6j(
        match[j1], match[j2], match[j12], match[j3], match[j], match[j23]
    )


# Utility constants.

_UNITY = Integer(1)
_NEG_UNITY = Integer(-1)
