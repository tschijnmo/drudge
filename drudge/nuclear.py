"""Utilities for nuclear problems."""

import functools
import re

from sympy import (
    Symbol, Function, Sum, symbols, Wild, KroneckerDelta, IndexedBase
)
from sympy.physics.quantum.cg import CG, Wigner3j, Wigner6j, Wigner9j, cg_simp

from .drudge import Tensor
from .fock import BogoliubovDrudge
from .term import Range


class j_(Function):
    """Symbolic access of j quantum number of an orbit."""
    pass


class tilde_(Function):
    """Symbolic access of the tilde part of an orbit."""
    pass


class m_(Function):
    """Symbolic access of the m quantum number of an orbit."""
    pass


_SUFFIXED = re.compile(r'^([a-zA-Z]+)([0-9]+)$')


def _decor_base(symb: Symbol, op):
    """Decorate the base part of the given symbol.

    The given symbol must have a digits-suffixed name, then the given
    operation will be applied to the base part, and recombined with the
    suffix to form the resulted symbol.
    """
    m = _SUFFIXED.match(symb.name)
    if not m:
        raise ValueError('Invalid symbol name to parse', symb)

    name, suffix = m.groups()
    return Symbol(op(name) + suffix)


def form_tilde(orig: Symbol):
    """Form the tilde symbol for a given orbit symbol.
    """
    return _decor_base(orig, lambda x: x + 'tilde')


def form_m(orig: Symbol):
    """Form the symbol for m quantum number for a given orbit symbol.
    """
    return _decor_base(orig, lambda _: 'm')


class NuclearBogoliubovDrudge(BogoliubovDrudge):
    """Utility drudge for nuclear theories based on Bogoliubov transformation.

    Different from the base :py:class:`BogoliubovDrudge` class, which
    concentrates on the transformation and the commutation rules, here we have
    more utility around the specifics about the nuclear Hamiltonian, especially
    the spherical symmetry.

    """

    def __init__(
            self, ctx, j_acc=j_, tilde_acc=tilde_, m_acc=m_,
            coll_j_range=Range('J', 0, Symbol('Jmax') + 1),
            coll_m_range=Range('M'),
            coll_j_dumms=tuple(Symbol('J{}'.format(i)) for i in range(10)),
            coll_m_dumms=tuple(Symbol('M{}'.format(i)) for i in range(10)),
            tilde_range=Range(r'\tilde(Q)'), form_tilde=form_tilde,
            m_range=Range('m'), form_m=form_m, **kwargs
    ):
        """Initialize the drudge object."""
        super().__init__(ctx, **kwargs)

        self.j_acc = j_acc
        self.tilde_acc = tilde_acc
        self.m_acc = m_acc
        self.set_name(
            j_=j_acc, tilde_=tilde_acc, m_=m_acc
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

        self.set_name(
            CG=CG, Wigner3j=Wigner3j, Wigner6j=Wigner6j, Wigner9j=Wigner9j
        )

        # For angular momentum decoupling.
        #
        # Here is a dictionary giving a callable for all implemented
        # decoupling.
        self._angdec_funcs = {}

        # Cases with total order of 4.
        qp_dumms = self.qp_dumms
        cj = coll_j_dumms[0]
        cm = coll_m_dumms[0]
        jkt1, jkt2, jkt3, jkt4 = [j_acc(tilde_acc(i)) for i in qp_dumms[:4]]
        mk2, mk3 = [m_acc(i) for i in [
            qp_dumms[2], qp_dumms[4]
        ]]
        order_4_cases = {
            (0, 4): (-1 ** (jkt1 + jkt2 + cj + cm), {0, 1}),
            (1, 3): (-1 ** (jkt2 - mk2 + cj), {1}),
            (2, 2): (1, {}),
            (3, 1): (-1 ** (jkt3 - mk3 + cj), {2}),
            (4, 0): (-1 ** (jkt3 + jkt4 + cj + cm), {2, 3}),
        }
        for k, v in order_4_cases.items():
            self._angdec_funcs[k] = functools.partial(
                self._form_angdec_def_4, v[0], v[1]
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
        kts = [self.tilde_acc(i) for i in ks]
        kt1, kt2, kt3, kt4 = kts
        jkt1, jkt2, jkt3, jkt4 = [self.j_acc(i) for i in kts]
        # Here the m's already includes the phase.
        mk1, mk2, mk3, mk4 = [
            self.m_acc(ks[i]) * (-1 if i in neg_ms else 1)
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
        tilde_acc = self.tilde_acc
        j_acc = self.j_acc
        m_range = self.m_range
        form_m = self.form_m
        m_acc = self.m_acc

        def expand(dumm: Symbol):
            """Expand a summation over quasi-particle orbitals."""
            tilde = tilde_acc(dumm)
            jtilde = j_acc(tilde)
            return [
                (form_tilde(dumm), tilde_range, tilde),
                (form_m(dumm), m_range[-jtilde, jtilde + 1], m_acc(dumm))
            ]

        return substed.expand_sums(self.qp_range, expand)

    @staticmethod
    def simplify_amp_sum(expr: Sum):
        """Attempt to simplify amplitude sums for nuclear problems.

        Here we specially concentrate on the simplification involving
        Clebosch-Gordan coefficients.
        """

        # TODO: Add more simplifications here.
        attempts = [
            cg_simp,
        ]
        for attempt in attempts:
            res = attempt(expr)
            if res is not None and res != expr:
                return res
            continue

        return None

