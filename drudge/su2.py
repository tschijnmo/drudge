"""Drudge for SU(2) Lie algebra."""

from sympy import Integer

from .genquad import GenQuadLatticeDrudge
from .term import Vec


class SU2LatticeDrudge(GenQuadLatticeDrudge):
    r"""Drudge for a lattice of su(2) algebras.

    This drudge has the commutation rules for :math:`\mathfrak{su}(2)` algebras
    in Cartan-Weyl form (Ladder operators).  Here both the shift and Cartan
    operators can have additional *lattice indices*.  Operators on different
    lattice sites always commute.  In detail, with a generator of the Cartan
    subalgebra denoted :math:`h`, and its raising and lowering operators by
    :math:`e` and :math:`f`, the commutation rules among the generators with the
    same index can be summarized as,

    .. math::

        [h, e] &= root \cdot e \\
        [f, e] &= -norm \cdot h - trail \\
        [f, h] &= root \cdot f \\

    where :math:`root`, :math:`norm`, and :math:`trail` are all tunable.  This
    is a slight generalization of the common Serre relations.

    Parameters
    ----------

    ctx
        The Spark context for the drudge.

    cartan
        The basis operator for the Cartan subalgebra (:math:`J^z` operator
        for spin problem).  It is registered in the name archive by the
        first letter in its label followed by an underscore.

    raise_
        The raising operator.  It is also also registered in the name
        archive by the first letter in its label followed by ``_p``.

    lower
        The lowering operator, registered by the first letter followed by
        ``_m``.

    root
        The coefficient for the commutator between the Cartan and shift
        operators.

    norm
        The coefficient for the commutator between the raising and lowering
        operators.

    trail
        A trailing scalar to be added to the Cartan generator in the
        commutator between the raising and lowering operators.

    order
        The normal order for the generators.  By default, the the
        normal-ordering operation would try to put raising operators before the
        Cartan operators, which come before the lowering operators.

    kwargs
        All other keyword arguments are given to the base class
        :py:class:`GenQuadDrudge`.

    """

    DEFAULT_CARTAN = Vec('J^z')
    DEFAULT_RAISE = Vec('J^+')
    DEFAULT_LOWER = Vec('J^-')

    def __init__(
            self, ctx, cartan=DEFAULT_CARTAN, raise_=DEFAULT_RAISE,
            lower=DEFAULT_LOWER, root=Integer(1), norm=Integer(2),
            trail=Integer(0), order=None, **kwargs
    ):
        r"""Initialize the drudge.
        """

        order = order if order is not None else (
            raise_, cartan, lower
        )
        comms = {
            (cartan, raise_): root * raise_,
            (cartan, lower): -root * lower,
            (raise_, lower): norm * cartan if trail == 0
            else norm * cartan + trail
        }

        super().__init__(ctx, order, comms, **kwargs)

        self.cartan = cartan
        self.raise_ = raise_
        self.lower = lower
        self.set_name(**{
            cartan.label[0] + '_': cartan,
            raise_.label[0] + '_p': raise_,
            lower.label[0] + '_m': lower
        })

    _latex_vec_mul = ' '
