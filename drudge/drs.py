"""Support for drudge scripts."""

import ast
import collections

from sympy import Symbol, Indexed, IndexedBase


#
# Special classes for SymPy objects
# ---------------------------------
#


class _Definable:
    """Mixin for definable objects in drudge scripts.
    """

    def def_as(self, rhs):
        """Define itself as a tensor.

        The definition is also added to the name archive.
        """
        drudge = self._drudge
        def_ = drudge.def_(self, rhs)
        drudge.set_name(def_)
        return def_

    def __le__(self, rhs):
        """Make a definition without touching the name archive."""
        return self._drudge.def_(self, rhs)


class DrsSymbol(_Definable, Symbol):
    """Symbols used in drudge scripts.

    The drudge symbol needs to behave as similar to the actual symbol as
    possible, because it is possible that they are used for keys in
    dictionaries.
    """

    __slots__ = [
        '_drudge',
        '_orig'
    ]

    def __new__(cls, drudge, name):
        """Create a symbol object."""
        symb = super().__new__(cls, name)
        return symb

    def __init__(self, drudge, name):
        """Initialize the symbol object."""
        self._drudge = drudge
        self._orig = Symbol(name)

    def __eq__(self, other):
        """Make equality comparison."""
        return self._orig == other

    def __hash__(self):
        """Compute the hash."""
        return hash(self._orig)

    def _hashable_content(self):
        """Hashable content for SymPy usages."""
        return self._orig._hashable_content()

    @classmethod
    def class_key(cls):
        return Symbol.class_key()

    def __getitem__(self, indices):
        """Index the given symbol.

        In drudge scripts, all symbols are by itself indexed bases.
        """
        base = IndexedBase(self._orig)
        if isinstance(indices, collections.Sequence):
            return DrsIndexed(self._drudge, base, *indices)
        else:
            return DrsIndexed(self._drudge, base, indices)

    def __iter__(self):
        """Disable iterability of the symbol.

        Or a default implementation from ``__getitem__`` will be used,
        which makes the symbols unable to be used as subscripts for indexed
        objects.
        """
        raise TypeError('Drudge script symbol cannot be iterated over.')


class DrsIndexed(_Definable, Indexed):
    """Indexed objects for drudge scripts."""

    __slots__ = [
        '_drudge',
        '_orig'
    ]

    def __new__(cls, drudge, base, *args, **kwargs):
        """Create an indexed object for drudge scripts."""
        indexed = super().__new__(cls, base, *args, **kwargs)
        return indexed

    def __init__(self, drudge, base, *args, **kwargs):
        """Initialize the indexed object."""
        self._drudge = drudge
        self._orig = Indexed(base, *args, **kwargs)

    def __eq__(self, other):
        """Make equality comparison."""
        return self._orig == other

    def __hash__(self):
        """Compute the hash."""
        return hash(self._orig)

    def _hashable_content(self):
        """Hashable content for SymPy usages."""
        return self._orig._hashable_content()

    @classmethod
    def class_key(cls):
        return Indexed.class_key()


#
# Python syntax tree manipulation
# -------------------------------
#


class _NumFixer(ast.NodeTransformer):
    """Fixer for number literals.

    Integer literals will be changed into creation of symbolic integers.
    """

    def visit_Num(self, node: ast.Num):
        """Update the number nodes."""
        val = node.n
        if isinstance(val, int):
            constr = ast.Name(id='Integer', ctx=ast.Load())
            ast.copy_location(constr, node)
            fixed = ast.Call(func=constr, args=[node], keywords=[])
            ast.copy_location(fixed, node)
            return fixed
        else:
            return val


_DEF_METH_NAME = 'def_as'


class _DefFixer(ast.NodeTransformer):
    """Fixer for tensor definitions.

    All augmented assignments ``<<=`` will be replaced by a call to the
    method with its name given in ``_DEF_METH_NAME``.
    """

    def visit_AugAssign(self, node: ast.AugAssign):
        """Update L-shift assignments."""
        op = node.op
        if not isinstance(op, ast.LShift):
            return node

        lhs = node.target
        if hasattr(lhs, 'ctx'):
            lhs.ctx = ast.Load()
        rhs = node.value

        deleg = ast.Attribute(
            value=lhs, attr=_DEF_METH_NAME, ctx=ast.Load()
        )
        ast.copy_location(deleg, lhs)
        call = ast.Call(func=deleg, args=[rhs], keywords=[])
        ast.copy_location(call, node)

        expr = ast.Expr(value=call)
        ast.copy_location(expr, node)
        return expr


_FIXERS = [_NumFixer(), _DefFixer()]


def compile_drs(src, filename):
    """Compile the drudge script."""

    root = ast.parse(src, filename=filename)

    for i in _FIXERS:
        root = i.visit(root)
        continue

    return compile(root, filename, mode='exec')
