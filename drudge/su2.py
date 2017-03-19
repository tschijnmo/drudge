"""Drudge for SU(2) Lie algebra."""

import functools
import operator

from sympy import Integer

from .genquad import GenQuadDrudge
from .term import Vec
from .utils import EnumSymbs, sympy_key


class RaisingLoweringChar(EnumSymbs):
    """Raising/lowering characters of ladder operators.

    Its values, which can be accessed as the class attributes ``RAISE`` and
    ``LOWER`` are also forwarded to module scope.  They should be used as the
    first index to vectors representing spin ladder operators.
    """

    _symbs_ = [
        ('RAISE', '+'),
        ('LOWER', '-')
    ]


RAISE = RaisingLoweringChar.RAISE
LOWER = RaisingLoweringChar.LOWER


class SU2LatticeDrudge(GenQuadDrudge):
    """Drudge for a lattice of SU(2) algebras.

    This drudge has the commutation rules for SU(2) algebras in Cartan-Weyl form
    (Ladder operators).  Here both the shift and Cartan operators can have
    additional *lattice indices*.  Operators on different lattice sites always
    commute.

    The the normal-ordering operation would try to put raising operators before
    the Cartan operators, which comes before the lowering operators.

    """

    def __init__(
            self, ctx, cartan_label='J^z', shift_label='J',
            root=Integer(1), norm=Integer(2),
            **kwargs
    ):
        """Initialize the drudge.

        Parameters
        ----------

        ctx
            The Spark context for the drudge.

        cartan_label
            The label for the basis operator in the Cartan subalgebra
            (:math:`J^z` operator).  It is registered in the name archive by the
            first letter followed by an underscore.

        shift_label
            The label for the shift (raising and lowering) operators.  They are
            also registered in the name archive by this label followed by ``_p``
            and ``_m``.

        root
            The coefficient for the commutator between the Cartan and shift
            operators.

        norm
            The coefficient for the commutator between the raising and lowering
            operators.

        All other keyword arguments are given to the base class
        :py:class:`GenQuadDrudge`.

        """
        super().__init__(ctx, **kwargs)

        cartan = Vec(cartan_label)
        self.cartan = cartan
        shift = Vec(shift_label)
        self.shift = shift
        raise_ = shift[RAISE]
        self.raise_ = raise_
        lower = shift[LOWER]
        self.lower = lower
        self.set_name(**{
            cartan_label[0] + '_': cartan,
            shift_label + '_p': raise_,
            shift_label + '_m': lower
        })

        self._root = root
        self._norm = norm

        self._swapper = functools.partial(
            _swap_su2, cartan=cartan, shift=shift, root=root, norm=norm
        )

    def swapper(self) -> GenQuadDrudge.Swapper:
        """The swapper for the spin algebra."""
        return self._swapper


def _swap_su2(vec1: Vec, vec2: Vec, *, cartan, shift, root, norm):
    """Swap two vectors based on the SU2 rules.
    """

    char1, indice1 = _parse_vec(vec1, cartan, shift)
    char2, indice2 = _parse_vec(vec2, cartan, shift)
    if len(indice1) != len(indice2):
        raise ValueError(
            'Invalid SU2 generators on lattice', (vec1, vec2),
            'incompatible number of lattice indices'
        )

    key1 = [sympy_key(i) for i in indice1]
    key2 = [sympy_key(i) for i in indice2]
    delta = functools.reduce(
        operator.mul, zip(indice1, indice2), _UNITY
    )

    if char1 == _RAISE:

        if char2 == _RAISE:
            if key1 < key2:
                return None
            elif key1 > key2:
                return _UNITY, _NOUGHT
            else:
                return _NOUGHT, _NOUGHT  # They are nilpotent.
        else:
            return None

    elif char1 == _CARTAN:

        if char2 == _RAISE:
            return _UNITY, root * delta * vec2
        elif char2 == _CARTAN:
            if key1 > key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None

    elif char1 == _LOWER:

        if char2 == _RAISE:
            return _UNITY, - norm * delta * cartan[indice1]
        elif char2 == _CARTAN:
            return _UNITY, root * delta * vec1
        else:
            if key1 < key2:
                return None
            elif key1 > key2:
                return _UNITY, _NOUGHT
            else:
                return _NOUGHT, _NOUGHT

    else:
        assert False


_RAISE = 0
_CARTAN = 1
_LOWER = 2


def _parse_vec(vec, cartan, shift):
    """Get the character and lattice indices of the vector.
    """

    if vec.base == cartan:
        return _CARTAN, vec.indices
    elif vec.base == shift:

        if len(vec.indices) < 1:
            raise ValueError(
                'Invalid shift operator for SU2 algebra', vec,
                'expecting raising/lowering character'
            )

        char_index = vec.indices[0]
        if char_index == RAISE:
            char = _RAISE
        elif char_index == LOWER:
            char = _LOWER
        else:
            raise ValueError(
                'Invalid shift character', char_index, 'on vector', vec
            )

        return char, vec.indices[1:]

    else:
        raise ValueError('Unexpected vector for SU2 algebra', vec)


_UNITY = Integer(1)
_NOUGHT = Integer(0)
