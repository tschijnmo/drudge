"""Support for drudge scripts."""

import argparse
import ast
import collections
import inspect

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

    def __call__(self, *args, **kwargs):
        """Make a call to a tensor method."""
        name = self.name

        if len(args) == 0:
            raise NameError('Undefined function', name)
        else:
            target = args[0]
            rest = args[1:]
            err = NameError('Invalid method', name, 'for', str(type(target)))
            if not hasattr(target, name):
                raise err
            meth = getattr(target, name)
            if inspect.ismethod(meth):
                # It has a caveat that static methods might not be able to be
                # called.
                return meth(*rest, **kwargs)
            elif len(rest) == 0 and len(kwargs) == 0:
                return meth
            else:
                raise err

    # Pickle support.

    def __getnewargs__(self):
        """Get the arguments for __new__."""
        return None, self.name

    def __getstate__(self):
        """Get the state for pickling."""
        return None

    def __setstate__(self, state):
        """Set the state according to pickled content."""
        from .drudge import current_drudge
        if current_drudge is None:
            raise ValueError(_PICKLE_ENV_ERR)
        self.__init__(current_drudge, self.name)


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

    def __getnewargs__(self):
        """Get the arguments for __new__."""
        return (None, self.base) + self.indices

    def __getstate__(self):
        """Get the state for pickling."""
        return None

    def __setstate__(self, state):
        """Set the state according to pickled content."""
        from .drudge import current_drudge
        if current_drudge is None:
            raise ValueError(_PICKLE_ENV_ERR)
        self.__init__(current_drudge, self.base, *self.indices)


_PICKLE_ENV_ERR = '''
Failed to unpickle,
not inside a pickling environment from pickle_env or inside a drudge script.
'''


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


#
# Execution environment
# ---------------------
#


class DrsEnv(dict):
    """The global scope for drudge script execution.
    """

    def __init__(self, dr, specials=None):
        """Initialize the scope."""

        super().__init__()

        self._drudge = dr
        path = [dr.names]
        self._path = path

        if specials is not None:
            path.append(specials)

        path.append(dr)
        import drudge
        path.append(drudge)

        try:
            import gristmill
        except ModuleNotFoundError:
            pass
        else:
            path.append(gristmill)

        import sympy
        path.append(sympy)

        import builtins
        path.append(builtins)

    def __missing__(self, key: str):
        """Get the missing name.

        The missing name will be returned.  The result is not cached to avoid
        surprise.
        """

        if key.startswith('__') and key.endswith('__'):
            raise KeyError(key)

        for i in self._path:
            if hasattr(i, key):
                resolv = getattr(i, key)
                break
            else:
                continue
        else:
            resolv = DrsSymbol(self._drudge, key)
        return resolv


#
# The main driver.
#


_DRUDGE_MAGIC = 'DRUDGE'

_CONF_HELP = '''
The config file for the drudge to be used, it needs to be a Python script
finally setting a global variable named ``{}`` for the drudge.
'''.format(_DRUDGE_MAGIC)


def main(argv=None):
    """The main driver for using drudge as a program.
    """

    parser = argparse.ArgumentParser(prog='drudge')
    parser.add_argument('conf', type=str, metavar='CONF', help=_CONF_HELP)
    parser.add_argument(
        'script', type=str, metavar='SCRIPT',
        help='The drudge script to execute'
    )
    if argv is not None:
        args = parser.parse_args(args=argv)
    else:
        args = parser.parse_args()

    with open(args.conf, 'r') as conf_fp:
        conf_src = conf_fp.read()
        conf_code = compile(conf_src, args.conf, 'exec')

    with open(args.script, 'r') as script_fp:
        script_src = script_fp.read()

    conf_env = {}
    exec(conf_code, conf_env)
    if _DRUDGE_MAGIC not in conf_env:
        raise ValueError('Drudge is not set to {} by {}'.format(
            _DRUDGE_MAGIC, args.conf
        ))
    drudge = conf_env[_DRUDGE_MAGIC]
    from drudge import Drudge
    if not isinstance(drudge, Drudge):
        raise ValueError('Invalid drudge is set to {} by {}'.format(
            _DRUDGE_MAGIC, args.conf
        ))

    env = drudge.exec_drs(script_src, args.script)
    return 0 if argv is None else env
