"""
Drudge, a symbolic system for non-commutative and tensor algebra
================================================================

"""

from .canonpy import Perm, Group
from .term import Range, Vec, Term
from .canon import IDENT, NEG, CONJ
from .drudge import Tensor, Drudge
from .wick import WickDrudge
from .fock import (
    CR, AN, FERMI, BOSE, FockDrudge, GenMBDrudge, PartHoleDrudge,
    UP, DOWN, SpinOneHalfGenDrudge, SpinOneHalfPartHoleDrudge
)
from .utils import sum_, prod_

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

    # Small user utilities.
    'sum_',
    'prod_'
]
