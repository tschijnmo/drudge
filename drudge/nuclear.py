"""Utilities for nuclear problems."""

import re

from sympy import Symbol, Function, Sum
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
            coll_m_range=Range('M', -Symbol('Jmax'), Symbol('Jmax') + 1),
            coll_j_dumms=tuple(Symbol('J{}'.format(i)) for i in range(10)),
            coll_m_dumms=tuple(Symbol('M{}'.format(i)) for i in range(10)),
            tilde_range=Range(r'\tilde(Q)'), form_tilde=form_tilde,
            m_range=Range('m', 0, Symbol('jmax') + 1), form_m=form_m,
            **kwargs
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

        self.set_tensor_method('decouple_qp_sums', self.decouple_qp_sums)

    def decouple_qp_sums(self, tensor: Tensor):
        """Decouple quasi-particle summation into the tilde and m parts.
        """

        # Cache as locals for Spark serialization.
        tilde_range = self.tilde_range
        form_tilde = self.form_tilde
        tilde_acc = self.tilde_acc
        m_range = self.m_range
        form_m = self.form_m
        m_acc = self.m_acc

        return tensor.expand_sums(self.qp_range, lambda x: [
            (form_tilde(x), tilde_range, tilde_acc(x)),
            (form_m(x), m_range, m_acc(x))
        ])

    @staticmethod
    def simplify_amp_sum(expr: Sum):
        """Attempt to simplify amplitude sums for nuclear problems.

        Here we specially concentrate on the simplification involving
        Clebosch-Gordan coefficients.
        """

        # TODO: Add more simplifications here.
        return cg_simp(expr)
