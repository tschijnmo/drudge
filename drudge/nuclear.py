"""Utilities for nuclear problems."""

import functools
import itertools
import re
import typing

import collections
from sympy import (
    Symbol, Function, Sum, symbols, Wild, KroneckerDelta, IndexedBase, Integer,
    sqrt, factor, Mul, Expr, Matrix, Add, Number, Pow
)
from sympy.physics.quantum.cg import CG, Wigner3j, Wigner6j, Wigner9j

from .drudge import Tensor
from .fock import BogoliubovDrudge
from .term import Range
from .utils import sympy_key, BCastVar

# Utility constants.

_UNITY = Integer(1)
_NEG_UNITY = Integer(-1)


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
    return _decor_base(orig, lambda _: 'm')


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
                Symbol('J{}'.format(i)) for i in range(1, 30)
            ),
            coll_m_dumms=tuple(
                Symbol('M{}'.format(i)) for i in range(1, 30)
            ),
            tilde_range=Range(r'\tilde{Q}', 0, Symbol('Ntilde')),
            form_tilde=form_tilde,
            m_range=Range('m'), form_m=form_m, **kwargs
    ):
        """Initialize the drudge object."""
        super().__init__(ctx, **kwargs)

        # Convenient names for quantum number access functions inside drudge
        # scripts.
        self.set_name(
            n_=NOf, NOf=NOf, l_=LOf, LOf=LOf, j_=JOf, JOf=JOf,
            tilde_=TildeOf, TildeOf=TildeOf, m_=MOf, MOf=MOf,
            pi_=PiOf, PiOf=PiOf
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

        self._am_sum_simplifiers = BCastVar(self.ctx, {
            # TODO: Add more simplifications here.
            2: [_sum_2_3j_to_delta],
            5: [_sum_4_3j_to_6j]
        })
        self.set_tensor_method('simplify_am', self.simplify_am)

        # All expressions for J/j, for merging of simple terms with factors in
        # J/j-hat style.
        self._j_exprs = frozenset(itertools.chain(self.coll_j_dumms, (
            JOf(i) for i in self.tilde_dumms
        )))

        # For angular momentum coupling.
        self.set_tensor_method('do_amc', self.do_amc)

        self.set_tensor_method('deep_simplify', self.deep_simplify)
        self.set_tensor_method('merge_j', self.merge_j)

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

    def simplify_am(self, tensor: Tensor):
        """Simplify the tensor for quantities related to angular momentum.

        Here we specially concentrate on the simplification involving scalar
        quantities related to angular momentum, like the Clebsch-Gordan
        coefficients.  For such simplifications, we need to invoke it explicitly
        or by the :py:meth:`deep_simplify` driver function, since it will not be
        called automatically during the default simplification for performance
        reasons.

        Note that this function will rewrite all CG coefficients in terms of
        Wigner 3j symbols even if no simplification comes from this.

        """

        tensor = tensor.map2amps(_rewrite_cg)

        # Initial simplification of some summations.
        tensor = tensor.simplify_sums(
            simplifiers=self._am_sum_simplifiers.bcast
        )

        # Deltas could come from some simplification rules.
        tensor = tensor.simplify_deltas()

        # Some summations could become simplifiable after the delta resolution.
        tensor = tensor.simplify_sums()

        return tensor

    def merge_j(self, tensor: Tensor):
        """Merge terms differing only in J/j factors.

        Frequently, from simplification of symbolic angular momentum quantities,
        we get factors like :math:`(2 j + 1)`.  Such factors could be expanded
        during the simplification, which may make the result longer and less
        clear with its physical meaning.  By this utility, such terms differing
        only by J/j factors will be attempted to be merged.
        """
        return tensor.merge(consts=self._j_exprs).map2amps(factor)

    def deep_simplify(self, tensor: Tensor):
        """Simplify the given tensor deeply.

        This driver function attempts to perform simplifications relevant to
        nuclear problems deeply.  It can be used in cases where all
        simplification attempts would like to be used.  When only some
        simplifications are needed, the individual simplification should better
        be invoked for better performance.
        """

        return (
            tensor.simplify()  # Initial simplification.
                .merge_j()  # This can possibly reduce the number of terms.
                .simplify_am()
                .simplify().merge_j()  # Final trial.
        )


#
# Angular momentum quantities simplification.
#

#
# Some utilities for J/M pairs in angular momentum quantities.  Note that here a
# bare m symbol is defined to be a symbol appearing as the m quantum number that
# is either an atomic symbol or a negation.  A bare m symbol is called an bare m
# dummy when it is also summed.
#
# Due to their presence in a lot of simplifications, here some work is given for
# their special treatment.
#

class _JM:
    """A pair of j and m quantum numbers.

    Attributes
    ----------

    j
        The j part.

    m
        The m part.

    """

    __slots__ = [
        'j',
        '_m',
        '_m_symb',
        '_m_phase'
    ]

    def __init__(self, j: Expr, m: Expr):
        """Initialize the pair."""
        self.j = j

        # For linters.
        self._m = None
        self._m_symb = None
        self._m_phase = None

        self.m = m

    @property
    def m(self):
        """The m part."""
        return self._m

    @m.setter
    def m(self, new_val):
        """Set new value for m."""

        self._m = new_val
        self._m_symb = None
        self._m_phase = None

        coeff, m_symb = _parse_linear(new_val)
        if m_symb is not None and coeff == 1 or coeff == -1:
            self._m_symb = m_symb
            self._m_phase = coeff

    @property
    def m_symb(self):
        """The only symbol in the m part when it is a bare symbol.
        """
        return self._m_symb

    @property
    def m_phase(self):
        """The phase for the bare symbol in the m part, when applicable.
        """
        return self._m_phase

    def inv_m(self):
        """Invert the sign of the m part.
        """
        self.m = -self.m

    def __repr__(self):
        """Get a simple string form for debugging.
        """
        return '{!s}, {!s}'.format(self.j, self.m)


#
# General utilities
#

def _parse_sum(expr: Sum):
    """Parse a SymPy summation into the summand and the summations.

    The summations are given as a hash set indexed by the dummies.
    """
    assert isinstance(expr, Sum)
    args = expr.args
    return args[0], {
        i: (j, k) for i, j, k in args[1:]
    }


class _UnexpectedForm(ValueError):
    """Exceptions for unexpected mathematical forms encountered.

    This exception can be raised during the matching of a mathematical
    expression against a pattern when it is found that the expression cannot
    possibly have a compliant form.  With this, each simplification attempt
    can have a central handling for noncompliance.

    """
    pass


def _fail(raise_):
    """Mark the failure of a pattern matching attempt.

    This utility can be used in functions which can be tuned to raise exception
    or return a None.  Note that it must be used together with return.
    """
    if raise_:
        raise _UnexpectedForm()
    return None


def _parse_linear(expr: Expr, symb_only=False):
    """Parse a linear function over a single symbolic quantity.

    When the expression is a number already, the symbolic part will be returned
    as None.  When the expression is not of the form of a number times a simple
    symbolic part, a pair of none will be returned.

    When symb_only is turned on, the symbolic part has to be an atomic symbol.

    """

    coeff, factors = expr.as_coeff_mul()

    if len(factors) == 0:
        return coeff, None
    elif len(factors) == 1:
        factor = factors[0]
        if symb_only and not isinstance(factor, Symbol):
            return None, None
        return coeff, factor
    else:
        return None, None


def _decomp_phase(phase: Expr, sums):
    """Decompose a phase into two factors according to dummy involvement.

    The phase is going to be decomposed into a product of two factors, with one
    having no reference to the symbols that is summed, and the other having the
    rest of the factors.

    In addition to the separation, the resulted two factors are also tried to be
    simplified.

    """

    expanded = phase.expand()
    dumms = sums.keys()

    inv = _UNITY
    other = _UNITY
    for i in (
            expanded.args if isinstance(expanded, Mul) else (expanded,)
    ):
        if i.atoms(Symbol).isdisjoint(dumms):
            other *= i
        else:
            inv *= i
        continue

    return tuple(
        i.powsimp().simplify() for i in (other, inv)
    )


def _rewrite_cg(expr):
    """Rewrite CG coefficients in terms of the Wigner 3j symbols.
    """
    j1, m1, j2, m2, j3, m3 = symbols(
        'j1 m1 j2 m2 j3 m3', cls=Wild
    )
    return expr.replace(
        CG(j1, m1, j2, m2, j3, m3),
        _NEG_UNITY ** (-j1 + j2 - m3) * sqrt(2 * j3 + 1) * Wigner3j(
            j1, m1, j2, m2, j3, -m3
        )
    )


#
# Special simplifications
#

class _Wigner3j:
    """Wrapper for a Wigner 3j symbols for easy manipulation.

    Parameters
    ----------

    expr
        The original Wigner 3j symbol as a SymPy expression.

    sums
        A container having the dummies as keys for the summations.  It is None
        by default, which disables the special treatment for m dummies.

    uniq_m
        If the bare m dummies are required to be unique within it.  When it is
        set to true, the m dummies mapping gives the slot index directly,
        otherwise it gives a set of indices.  If duplicated bare m dummies
        appears when it is set to true, exception is raised.

    Attributes
    ----------

    indices
        The list of the six indices to the Wigner 3j symbol, as j/m pairs.

    phase_decided
        If the phase of the m symbols are decided.

    """

    __slots__ = [
        'indices',
        '_slot_decided',
        'phase_decided',
        '_uniq_m',
        '_total_j',
        '_m_dumms'
    ]

    def __init__(self, expr: Wigner3j, sums=None, uniq_m=True):
        """Initialize the handler."""
        assert isinstance(expr, Wigner3j)

        args = expr.args
        indices = [
            _JM(args[i], args[i + 1]) for i in range(0, 6, 2)
        ]
        self.indices = indices
        self._slot_decided = [False for _ in range(3)]
        self.phase_decided = False

        if sums is not None:
            self._uniq_m = uniq_m
            m_dumms = {}
            self._m_dumms = m_dumms
            for i, v in enumerate(indices):
                m_symb = v.m_symb
                if m_symb is None or m_symb not in sums:
                    continue

                if m_symb in m_dumms:
                    if uniq_m:
                        raise _UnexpectedForm()
                    else:
                        m_dumms[m_symb].add(i)
                else:
                    m_dumms[m_symb] = i if uniq_m else {i}
                continue

        # This does not change after the manipulations.
        self._total_j = sum(i.j for i in indices)

    @property
    def m_dumms(self):
        """The bare m dummies in this symbol.

        A mapping from bare m dummies to the slot index for the j/m pair that it
        appears at as a bare m symbol.
        """
        return self._m_dumms

    def is_decided(self, slot_index):
        """If the content for a slot has been decided.
        """
        return self._slot_decided[slot_index]

    def decide(self, slot_index):
        """Mark a slot index as decided.

        Note that a slot can be marked as decided multiple times.
        """
        self._slot_decided[slot_index] = True

    def swap(
            self, src, dest, raise_src=False, raise_dest=False,
            raise_pred=False
    ) -> typing.Optional[Expr]:
        """Try to swap index slots.

        The source can be given either as an index or a predicate on _JM object.
        When a predicate is given, there should be only one slot satisfying the
        predicate.  The destination has to be given as an index.  After the
        swapping, the destination slot will be automatically marked as decided.

        For the action to be successful, both slots needs to be undecided.  The
        ``try_`` arguments controls if _UnexpectedForm is to be raised or a
        plain None is to be returned.  When the swapping is successful, the
        phase for that swap is returned.

        """

        if not isinstance(src, int):
            pred = src
            src = None  # None for not found, False for duplicated.
            for i, v in enumerate(self.indices):
                if pred(v):
                    src = i if src is None else False
                else:
                    continue
            if src is None or src is False:
                return _fail(raise_pred)

        if src == dest:
            self.decide(dest)
            return _UNITY

        for i, j in [(src, raise_src), (dest, raise_dest)]:
            if self.is_decided(i):
                return _fail(j)

        indices = self.indices
        src_jm, dest_jm = indices[src], indices[dest]
        # Core swapping.
        indices[src], indices[dest] = dest_jm, src_jm
        # Update bare m dummy mapping.
        src_m, dest_m = src_jm.m_symb, dest_jm.m_symb
        if src_m != dest_m:
            # When src_m and dest_m are both None, no need for any treatment.
            # Or when they are the same symbol, the set of appearances of that
            # symbol does not need to be changed as well.
            for m_symb, old_idx, new_idx in [
                (src_m, src, dest),
                (dest_m, dest, src)
            ]:
                if m_symb is None or m_symb not in self._m_dumms:
                    continue
                entry = self._m_dumms[m_symb]
                if self._uniq_m:
                    assert entry == old_idx
                    self._m_dumms[m_symb] = new_idx
                else:
                    entry.remove(old_idx)  # Key error when not present.
                    assert new_idx not in entry
                    entry.add(new_idx)

        self.decide(dest)
        return _NEG_UNITY ** self._total_j

    def inv_ms(self):
        """Invert the sign of all m quantum numbers.
        """
        for i in self.indices:
            i.inv_m()
            continue

        return _NEG_UNITY ** self._total_j

    def __repr__(self):
        """Form a string representation for easy debugging.
        """
        return '_Wigner3j({})'.format(', '.join([
            repr(i) for i in self.indices
        ]))


def _check_m_contr(
        factor1: _Wigner3j, factor2: _Wigner3j, decided_ms, normal, inv,
        raise_=True
) -> typing.Optional[Expr]:
    """Check if two Wigner 3j symbols contracts according to the given pattern.

    A (normal) contraction is defined as a pair of slots in the two Wigner 3j
    factors with equal j parts and equal bare m parts that are summed over -j to
    j.  Or when the bare m parts are negation of each other, it will be called
    inverted contractions.

    Each normal/inverted contraction is given as a pair of slot indices.  Then
    this function will attempt to permute the indices to the factors to arrange
    them into the contraction pattern.  When it fails, a _UnexpectedForm will be
    thrown if ``raise_`` is turned on, otherwise a none will be returned.  When
    it succeeds, the phase factor that comes with the arrangement will be
    returned.

    This function also attempts to put the two m symbols for normal
    contractions, and the contracted m of the first Wigner 3j symbol in inverted
    contractions, to be in plain symbol form, without any negation.  The
    substitutions needed in the phase for this are added to the dictionary
    ``decided_ms`` in the argument.

    .. warning::

        Due to the naive handling of phases, usage of this function should
        gradually expand the tested part of the contraction graph, rather than
        starting with disconnected parts and them try to patch them together.

    TODO: Improve the interface of this function.

    """

    # Bare m dummies contracted between the two factors.
    shared_dumms = factor1.m_dumms.keys() & factor2.m_dumms.keys()
    if len(shared_dumms) != len(normal) + len(inv):
        return _fail(raise_)

    #
    # TODO: Add more pedantic checking
    #
    # For two m slots to be actually considered to be contracted together, the j
    # symbols above them need to be the same, and they need to be summed over
    # the correct range.  Here, due to some limitations, pedantic checking can
    # possibly block some important simplifications.  For instance, two
    # different j's above the two m's can actually be equal by a delta, which is
    # hard to recognized inside the current framework.  Hence pedantic checking
    # is temporarily skipped here.
    #
    # In actual rules, we also generally skip the checking of j and the range of
    # m as well for the same reason.
    #

    # Try two phases if possible.
    check = functools.partial(
        _check_m_contr_fixed_phase, factor1, factor2, normal, inv, shared_dumms
    )
    phase_undecided = [i for i in [factor1, factor2] if not i.phase_decided]
    if len(phase_undecided) == 0:
        res = check()
    else:
        res = check()
        if res is None:
            inv_phase = phase_undecided[0].inv_ms()
            res = check()
            if res is None:
                # Try restore original phase.
                phase_undecided[0].inv_ms()
            else:
                res *= inv_phase

    if res is None:
        return _fail(raise_)

    factor1.phase_decided = True
    factor2.phase_decided = True

    # Finalize the phase of the m dummies.
    for if_normal, slots in ((True, normal), (False, inv)):
        for i1, i2 in slots:
            assert factor1.is_decided(i1)
            assert factor2.is_decided(i2)
            # The contracted.
            index1 = factor1.indices[i1]
            index2 = factor2.indices[i2]
            assert index1.m_symb == index2.m_symb
            m_symb = index1.m_symb
            if m_symb in decided_ms:
                # Non-conventional contraction pattern where the dummy has
                # already been summed elsewhere.
                return _fail(raise_)

            if if_normal:
                if index1.m_phase == -1:
                    assert index2.m_phase == -1
                    assert index1.m == -m_symb
                    assert index2.m == -m_symb
                    index1.m = m_symb
                    index2.m = m_symb
                    decided_ms[m_symb] = -m_symb
                else:
                    assert index2.m_phase == 1
                    assert index1.m == m_symb
                    assert index2.m == m_symb
                    decided_ms[m_symb] = m_symb
            else:
                if index1.m_phase == -1:
                    assert index2.m_phase == 1
                    assert index1.m == -m_symb
                    assert index2.m == m_symb
                    index1.m = m_symb
                    index2.m = -m_symb
                    decided_ms[m_symb] = -m_symb
                else:
                    assert index2.m_phase == -1
                    assert index1.m == m_symb
                    assert index2.m == -m_symb
                    decided_ms[m_symb] = m_symb

    return res


def _check_m_contr_fixed_phase(
        factor1, factor2, normal, inv, shared_dumms
):
    """Check the m contraction of two symbols with fixed phase.
    """

    factors = [factor1, factor2]
    indices1 = factor1.indices
    indices2 = factor2.indices

    # Free dummies are dummies whose both appearance are on undecided slots.
    normal_dumms = set()
    free_normal_dumms = set()
    inv_dumms = set()
    free_inv_dumms = set()
    for dumm in shared_dumms:
        i1, i2 = [j.m_dumms[dumm] for j in factors]
        m1, m2 = indices1[i1].m, indices2[i2].m
        if m1 == m2:
            dumms = normal_dumms
            frees = free_normal_dumms
        elif m1 == -m2:
            dumms = inv_dumms
            frees = free_inv_dumms
        else:
            assert False
        dumms.add(dumm)
        if not factor1.is_decided(i1) and not factor2.is_decided(i2):
            frees.add(dumm)
        continue

    if len(normal_dumms) != len(normal) or len(inv_dumms) != len(inv):
        return None

    phase = _UNITY
    for to_proc, dumms, frees in [
        (normal, normal_dumms, free_normal_dumms),
        (inv, inv_dumms, free_inv_dumms)
    ]:
        for i1, i2 in to_proc:
            decided1 = factor1.is_decided(i1)
            decided2 = factor2.is_decided(i2)
            m1 = indices1[i1].m_symb
            m2 = indices2[i2].m_symb
            if m1 == m2 and m1 in dumms:
                # When we found the contraction is already satisfied.
                factor1.decide(i1)
                factor2.decide(i2)
                if m1 in frees:
                    frees.remove(m1)
            # Now the contraction must be currently unsatisfied.
            elif decided1 and decided2:
                return None
            elif decided1:
                # Try refill slot 2 to match the m in slot 1.
                if m1 not in dumms:
                    return None
                curr_i2 = factor2.m_dumms[m1]
                if factor2.is_decided(curr_i2):
                    return None
                phase *= factor2.swap(curr_i2, i2)
            elif decided2:
                # Try refill slot 1 to match the m in slot 2.
                if m2 not in dumms:
                    return None
                curr_i1 = factor1.m_dumms[m2]
                if factor1.is_decided(curr_i1):
                    return None
                phase *= factor1.swap(curr_i1, i1)
            else:
                # Try to move a free contraction to the two required slots.
                if len(frees) == 0:
                    return None
                # Try to get an m already touching.
                if indices1[i1].m_symb in frees:
                    new_m = indices1[i1].m_symb
                    frees.remove(new_m)
                elif indices2[i2].m_symb in frees:
                    new_m = indices2[i2].m_symb
                    frees.remove(new_m)
                else:
                    new_m = frees.pop()
                curr_i1 = factor1.m_dumms[new_m]
                curr_i2 = factor2.m_dumms[new_m]
                assert not factor1.is_decided(curr_i1)
                assert not factor2.is_decided(curr_i2)
                phase *= factor1.swap(curr_i1, i1)
                phase *= factor2.swap(curr_i2, i2)

            # Continue to next contraction.
            continue
        # Continue to inverted contractions.
        continue

    return phase


def _parse_3js(expr, **kwargs) -> typing.Tuple[
    Expr, typing.List[_Wigner3j]
]:
    """Parse an expression of Wigner 3j symbol product.

    The result is a phase and a list of wrappers of 3j symbols.  All keyword
    arguments are forwarded to the Wigner 3j wrapper.
    """

    factors = expr.args if isinstance(expr, Mul) else (expr,)
    phase = _UNITY
    wigner_3js = []
    for i in factors:
        if isinstance(i, Wigner3j):
            wigner_3js.append(_Wigner3j(i, **kwargs))
        else:
            phase *= i
        continue
    return phase, wigner_3js


def _sum_2_3j_to_delta(expr: Sum):
    """Attempt to sum two 3j symbols into deltas.

    The exact rule can be found at Wolfram_, Equations 1-3.  Here they are
    implemented as a unified rule.

    .. _Wolfram: http://functions.wolfram.com/HypergeometricFunctions
    /ThreeJSymbol/23/01/02/

    """

    if len(expr.args) != 3:
        return None
    summand, sums = _parse_sum(expr)

    try:
        phase, wigner_3js = _parse_3js(summand, sums=sums)
        if len(wigner_3js) != 2:
            return None

        decided_ms = {}
        phase *= _check_m_contr(wigner_3js[0], wigner_3js[1], decided_ms, [
            (0, 0), (1, 1)
        ], [])
    except _UnexpectedForm:
        return None

    noinv_phase, phase = _decomp_phase(phase.xreplace(decided_ms), sums)
    simpl = _Wigner3jMSimpl(wigner_3js, sums)
    simpl_phase = simpl.simplify(phase)
    if simpl_phase != 1:
        return None

    indices0 = wigner_3js[0].indices
    indices1 = wigner_3js[1].indices
    j1, m1 = indices0[0].j, indices0[0].m
    j2, m2 = indices0[1].j, indices0[1].m
    j3, m3 = indices0[2].j, indices0[2].m
    assert m1 == indices1[0].m
    assert m2 == indices1[1].m
    j4, m4 = indices1[2].j, indices1[2].m

    return KroneckerDelta(j3, j4) * KroneckerDelta(
        m3, m4
    ) / (2 * j3 + 1) * noinv_phase


def _sum_4_3j_to_6j(expr: Sum):
    """Attempt to sum four 3j symbols into a 6j symbol.

    The exact rule can be found at Wolfram_.

    .. _Wolfram: http://functions.wolfram.com/07.39.23.0014.01

    """

    if len(expr.args) != 6:
        return None

    summand, sums = _parse_sum(expr)

    # From here on, we gradually rewrite the expression into the pattern of the
    # rule.
    try:
        phase, wigner_3js = _parse_3js(summand, sums=sums)
        if len(wigner_3js) != 4:
            return None

        ext_3js = []
        int_3js = []
        for i in wigner_3js:
            if len(i.m_dumms) == 3:
                int_3js.append(i)
            elif len(i.m_dumms) == 2:
                ext_3js.append(i)
            else:
                return None
            continue
        if len(ext_3js) != 2 or len(int_3js) != 2:
            return None

        # Put the external ones in the middle.
        for ext_3j in ext_3js:
            phase *= ext_3j.swap(
                lambda x: x.m_symb not in sums, 1
            )
            continue

        decided_ms = {}

        # For performance.
        empty = []

        # Get the edge between the two internal 3js.
        phase *= _check_m_contr(
            int_3js[0], int_3js[1], decided_ms, empty, [(2, 2)]
        )

        # Match the corresponding slots in the pattern.
        phase *= _check_m_contr(
            ext_3js[0], int_3js[0], decided_ms, empty, [(2, 0)]
        )
        phase *= _check_m_contr(
            int_3js[0], ext_3js[1], decided_ms, empty, [(1, 0)]
        )
        phase *= _check_m_contr(
            ext_3js[1], int_3js[1], decided_ms, empty, [(2, 0)]
        )
        phase *= _check_m_contr(
            ext_3js[0], int_3js[1], decided_ms, empty, [(0, 1)]
        )
    except _UnexpectedForm:
        return None

    ext_0 = ext_3js[0].indices
    ext_1 = ext_3js[1].indices
    int_0 = int_3js[0].indices
    int_1 = int_3js[1].indices
    j2, m2 = ext_0[0].j, ext_0[0].m
    j3, m3 = ext_0[1].j, -ext_0[1].m
    j1, m1 = ext_0[2].j, ext_0[2].m
    assert m1 == -int_0[0].m
    j5, m5 = int_0[1].j, int_0[1].m
    j6, m6 = int_0[2].j, int_0[2].m
    assert m5 == -ext_1[0].m
    jprm3, mprm3 = ext_1[1].j, ext_1[1].m
    j4, m4 = ext_1[2].j, ext_1[2].m
    assert m4 == -int_1[0].m
    assert m2 == -int_1[1].m
    assert m6 == -int_1[2].m

    noinv_phase, phase = _decomp_phase(phase.xreplace(decided_ms), sums)
    simpl = _Wigner3jMSimpl(wigner_3js, sums)
    simpl_phase = simpl.simplify(phase)
    expected_phase = simpl.simplify((-1) ** (
            - m1 - m2 - m4 - m5 - m6
    ))
    if (simpl_phase / expected_phase).simplify() != 1:
        return None

    return (-1) ** (j3 - m3 - j1 - j2 - j4 - j5 - j6) / (2 * j3 + 1) * (
            KroneckerDelta(j3, jprm3)
            * KroneckerDelta(m3, mprm3)
            * Wigner6j(j1, j2, j3, j4, j5, j6)
    ) * noinv_phase


class _Wigner3jMSimpl:
    """Simplifier based the relations among m's of Wigner 3j symbols.

    The three m quantum numbers in a Wigner 3j symbol need to add to zero for it
    to be non-zero.  This can be used for the simplification of the factor ahead
    of a product of 3j symbols.

    This simplifier can be initialized with an iterable of 3j symbols, then it
    can be used to simplify factors multiplying their product.

    Parameters
    ----------

    wigner_3js
        The Wigner 3j symbols.

    sums
        When given, only relations among m symbols inside this container will be
        considered.  Or all relations from 3j symbols with bare m symbols will
        be considered.

    Notes
    -----

    Currently, only bare m symbols are considered.

    TODO: Make it applicable to more general forms of the m quantum number.

    """

    __slots__ = [
        '_symb2idx',
        '_rel_vecs'
    ]

    def __init__(self, wigner_3js: typing.Iterable[_Wigner3j], sums=None):
        """Initialize the simplifier.
        """

        symb2idx = collections.OrderedDict()

        rels = []  # Written as dictionary.
        for i in wigner_3js:
            rel = {}
            for j in i.indices:
                m_symb = j.m_symb
                if m_symb is None:
                    break
                if sums is not None and m_symb not in sums:
                    break

                if m_symb not in symb2idx:
                    symb2idx[m_symb] = len(symb2idx)
                rel[m_symb] = j.m_phase

            else:
                rels.append(rel)
            continue

        # Cast the relations into vectors.
        raw_rel_vecs = []
        n_symbs = len(symb2idx)
        for i in rels:
            coeffs = [0 for _ in range(n_symbs)]
            for k, v in i.items():
                coeffs[symb2idx[k]] = v
                continue
            raw_rel_vecs.append(Matrix(coeffs))
            continue

        # Gram-Schmidt procedure to make the relations orthogonal.
        rel_vecs = []
        for i in raw_rel_vecs:
            vec = _proj_out(rel_vecs, i)
            if not vec.is_zero:
                rel_vecs.append(vec)

        self._symb2idx = symb2idx
        self._rel_vecs = rel_vecs

    def simplify(self, expr: Expr):
        """Simplify the given expression given the m relations of 3j symbols.
        """
        expr = expr.powsimp().simplify()

        if not isinstance(expr, Pow):
            expr = expr.replace(Add, self._simpl_add)
        else:
            # Specialized treatment for the form we normally see in spherical
            # problems.  The above general treatment could skip the processing
            # when we have a simple exponential of a plain symbol
            base, expon = expr.args
            if isinstance(expon, Add):
                expon = self._simpl_add(*expon.args)
            else:
                # For a single term in the exponent.
                expon = self._simpl_add(expon)
            expr = base ** expon

        return expr.simplify()

    def _simpl_add(self, *args):
        """Simplify an addition.
        """

        # Separate the addends into the linear factors in the m symbols under
        # consideration, and others.
        other = 0
        coeffs = [0 for _ in self._symb2idx]

        symb2idx = self._symb2idx
        for arg in args:
            coeff, symb = _parse_linear(arg)
            if symb is None or symb not in symb2idx:
                other += arg
            else:
                coeffs[symb2idx[symb]] += coeff
            continue

        res = other
        simpl_coeffs = _proj_out(self._rel_vecs, Matrix(coeffs))
        for i, j in zip(symb2idx.keys(), simpl_coeffs):
            if j != 0:
                res += i * j

        return res


def _proj_out(bases, vec):
    """Project out the components of the vector on the given bases.

    The bases are assumed to be orthogonal, not necessarily normalized.
    """
    for i in bases:
        vec -= vec.project(i)
    return vec


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
