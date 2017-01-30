"""
Drudge, a symbolic system for non-commutative and tensor algebra
================================================================

"""

from .canonpy import Perm, Group
from .term import Range, Vec, Term
from .canon import IDENT, NEG, CONJ
from .drudge import Tensor, Drudge

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
]
