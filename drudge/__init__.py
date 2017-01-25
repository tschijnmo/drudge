"""
Drudge, a symbolic system for non-commutative and tensor algebra
================================================================

"""

from .canonpy import Perm, Group, canon_eldag
from .term import Range, Vec, Term

__all__ = [
    # Canonpy.
    'Perm',
    'Group',
    'canon_eldag',

    # Vec.
    'Vec',

    # Term.
    'Range',
    'Term'
]
