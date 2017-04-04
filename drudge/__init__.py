"""
Drudge, a symbolic system for non-commutative and tensor algebra
================================================================

"""

from .canonpy import Perm, Group
from .term import Range, Vec, Term
from .canon import IDENT, NEG, CONJ
from .drudge import Tensor, TensorDef, Drudge
from .wick import WickDrudge
from .fock import (
    CR, AN, FERMI, BOSE, FockDrudge, GenMBDrudge, PartHoleDrudge,
    UP, DOWN, SpinOneHalfGenDrudge, SpinOneHalfPartHoleDrudge,
    RestrictedPartHoleDrudge
)
from .genquad import GenQuadDrudge
from .su2 import SU2LatticeDrudge
from .clifford import CliffordDrudge, inner_by_delta
from .utils import sum_, prod_, Stopwatch

__version__ = '0.5.0'

__all__ = [
    # Canonpy.
    'Perm',
    'Group',

    # Vec.
    'Vec',

    # Term.
    'Range',
    'Term',

    # Canon.
    'IDENT',
    'NEG',
    'CONJ',

    # Drudge.
    'Tensor',
    'TensorDef',
    'Drudge',

    # Different problem-specific drudges.
    #
    # Base Wick algebra.
    'WickDrudge',

    # Many-body theories.
    'CR', 'AN', 'FERMI', 'BOSE',
    'FockDrudge',
    'GenMBDrudge', 'PartHoleDrudge',
    'UP', 'DOWN',
    'SpinOneHalfGenDrudge', 'SpinOneHalfPartHoleDrudge',
    'RestrictedPartHoleDrudge',

    # Other algebraic systems.
    'GenQuadDrudge',
    'SU2LatticeDrudge',
    'CliffordDrudge',
    'inner_by_delta',

    # Small user utilities.
    'sum_',
    'prod_',
    'Stopwatch'
]
