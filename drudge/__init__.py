"""
Drudge, a symbolic system for non-commutative and tensor algebra
================================================================

"""

from .canonpy import Perm, Group, canon_eldag
from .vec import Vec
from .term import Range

__all__ = [
    # Canonpy.
    'Perm',
    'Group',
    'canon_eldag',

    # Vec.
    'Vec',

    # Term.
    'Range'
]
