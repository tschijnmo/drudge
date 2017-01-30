"""
Drudge, a symbolic system for non-commutative and tensor algebra
================================================================

"""

from .term import Range, Vec, Term
from .canon import IDENT, NEG, CONJ
from .drudge import Tensor, Drudge

__all__ = [
    # Canonpy.
    'Perm',
    'Group',
    'canon_eldag',

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
