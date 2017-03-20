Drudge tutorial for beginners
=============================

.. currentmodule:: drudge


Get started
-----------


Drudge is a library built on top of the SymPy computer algebra library for
noncommutative and tensor alegbras.  Usually for these style of problems, the
symbolic manipulation and simplification of mathematical expressions requires a
lot of context-dependent information, like the specific commutation rules and
things like the dummy symbols to be used for different ranges.  So the primary
entry point for using the library is the :py:class:`Drudge` class, which serves
as a central repository of all kinds of domain-specific informations.  To
create a drudge instance, we need to give it a Spark context so that it is
capable of parallelize things.  For instance, to run things locally with all
available cores, we can do

.. doctest::

    >>> from pyspark import SparkContext  # doctest: +SKIP
    >>> spark_ctx = SparkContext('local[*]', 'drudge-tutorial')

For using Spark in cluster computing environment, please refer to the Spark
documentation and setting of your cluster.  With the spark context created, we
can make the main entry point for drudge,

.. doctest::

    >>> import drudge
    >>> dr = drudge.Drudge(spark_ctx)

Then from it, we can create the symbolic expressions as :py:class:`Tensor`
objects, which are basically mathematical expressions containing noncommutative
objects and symbolic summations.  For the noncommutativity, in spite of the
availability of some basic support of it in SymPy, here we have the
:py:class:`Vec` class to specifically designate the noncommutativity of its
multiplication.  It can be created with a label and indexed with SymPy
expressions.

.. doctest::

    >>> v = drudge.Vec('v')
    >>> import sympy
    >>> a = sympy.Symbol('a')
    >>> str(v[a])
    'v[a]'

For the symbolic summations, we have the :py:class:`Range` class, which denotes
a symbolic set that a variable could be summed over.  It can be created by just
a label.

.. doctest::

    >>> l = drudge.Range('L')

With these, we can create tensor objects by using the :py:meth:`Drudge.sum`
method,

.. doctest::

    >>> x = sympy.IndexedBase('x')
    >>> tensor = dr.sum((a, l), x[a] * v[a])
    >>> str(tensor)
    'sum_{a} x[a] * v[a]'

Now we got a symbolic tensor of a sum of vectors modulated by a SymPy
IndexedBase.  Actually any type of SymPy expression can be used to modulate the
noncommutative vectors.

.. doctest::

    >>> tensor = dr.sum((a, l), sympy.sin(a) * v[a])
    >>> str(tensor)
    'sum_{a} sin(a) * v[a]'

And we can also have multiple summations and product of the vectors.

.. doctest::

    >>> b = sympy.Symbol('b')
    >>> tensor = dr.sum((a, l), (b, l), x[a, b] * v[a] * v[b])
    >>> str(tensor)
    'sum_{a, b} x[a, b] * v[a] * v[b]'

Of cause the multiplication of the vectors will not be commutative,

.. doctest::

    >>> tensor = dr.sum((a, l), (b, l), x[a, b] * v[b] * v[a])
    >>> str(tensor)
    'sum_{a, b} x[a, b] * v[b] * v[a]'

Normally, for each symbolic range, we have some traditional symbols used as
dummies for summations over them, giving these information to drudge objects
can be very helpful.  Here in this demonstration, we can use the
:py:meth:`Drudge.set_dumms` method.

.. doctest::

    >>> dr.set_dumms(l, sympy.symbols('a b c d'))
    [a, b, c, d]
    >>> dr.add_resolver_for_dumms()

where the call to the :py:meth:`Drudge.add_resolver_for_dumms` method could
tell the drudge to interpret all the dummy symbols to be over the range that
they are set to.  By giving drudge object such domain-specific information, we
can have a lot convenience.  For instance, now we can use Einstein summation
convention to create tensor object, without the need to spell all the
summations out.

.. doctest::

    >>> tensor = dr.einst(x[a, b] * v[a] * v[b])
    >>> str(tensor)
    'sum_{a, b} x[a, b] * v[a] * v[b]'

Also the drudge knows what to do when more dummies are needed in mathematical
operations.  For instance, when we multiply things,

.. doctest::

    >>> tensor = dr.einst(x[a] * v[a])
    >>> prod = tensor * tensor
    >>> str(prod)
    'sum_{a, b} x[a]*x[b] * v[a] * v[b]'

Here the dummy :math:`b` is automatically used since the drudge object knows
available dummies for its range.  Also the range and the dummies are
automatically added to the name archive of the drudge, which can be access by
:py:attr:`Drudge.names`.

.. doctest::

    >>> p = dr.names
    >>> p.L
    Range('L')
    >>> p.L_dumms
    [a, b, c, d]
    >>> p.d
    d

Here in this example, we set the dummies ourselves by
:py:meth:`Drudge.set_dumms`.  Normally, in subclasses of :py:class:`Drudge` for
different specific problems, such setting up is already finished within the
class.  We can just directly get what we need from the names archive.  There is
also a method :py:meth:`Drudge.inject_names` for the convenience of interactive
work.


Tensor manipulations
--------------------


Now with tensors created by :py:meth:`Drudge.sum` or :py:meth:`Drudge.einst`, a
lot of mathematical operations are available to them.   In addition to the
above example of (noncommutative) multiplication, we can also have the linear
algebraic operations of addition and scalar multiplication.

.. doctest::

    >>> tensor = dr.einst(x[a] * v[a])
    >>> y = sympy.IndexedBase('y')
    >>> res = tensor + dr.einst(y[a] * v[a])
    >>> str(res)
    'sum_{a} x[a] * v[a]\n + sum_{a} y[a] * v[a]'

    >>> res = 2 * tensor
    >>> str(res)
    'sum_{a} 2*x[a] * v[a]'

We can also perform some complex substitutions on either the vector or the
amplitude part, by using the :py:meth:`Drudge.subst` method.

.. doctest::

    >>> t = sympy.IndexedBase('t')
    >>> w = drudge.Vec('w')
    >>> substed = tensor.subst(v[a], dr.einst(t[a, b] * w[b]))
    >>> str(substed)
    'sum_{a, b} x[a]*t[a, b] * w[b]'

    >>> substed = tensor.subst(x[a], sympy.sin(a))
    >>> str(substed)
    'sum_{a} sin(a) * v[a]'

Note that here the substituted vector does not have to match the left-hand side
of the substitution exactly, pattern matching is done here.  Other mathematical
operations are also available, like symbolic differentiation by
:py:meth:`Tensor.diff` and commutation by ``|`` operator
:py:meth:`Tensor.__or__`.

Tensors are purely mathematical expressions, while the utility class
:py:class:`TensorDef` can be construed as tensor expressions with a left-hand
side.  They can be easily created by :py:meth:`Drudge.define` and
:py:meth:`Drudge.define_einst`.

.. doctest::

    >>> v_def = dr.define_einst(v[a], t[a, b] * w[b])
    >>> str(v_def)
    'v[a] = sum_{b} t[a, b] * w[b]'

Their method :py:meth:`TensorDef.act` is like a active voice version of
:py:meth:`Tensor.subst` and could come handy when we need to substitute the same
definition in multiple inputs.

.. doctest::

    >>> res = v_def.act(tensor)
    >>> str(res)
    'sum_{a, b} x[a]*t[a, b] * w[b]'

More importantly, the definitions can be indexed directly, and the result is
designed to work well inside :py:meth:`Drudge.sum` or :py:meth:`Drudge.einst`.
For instance, for the same result, we could have,

.. doctest::

    >>> res = dr.einst(x[a] * v_def[a])
    >>> str(res)
    'sum_{b, a} x[a]*t[a, b] * w[b]'

When the only purpose of a vector or indexed base is to be substituted and we
never intend to write tensor expressions directly in terms of them, we can just
name the definition with a short name directly and put the actual base inside
only.  For instance,

.. doctest::

    >>> c = sympy.Symbol('c')
    >>> f = dr.define_einst(sympy.IndexedBase('f')[a, b], x[a, c] * y[c, b])
    >>> str(f)
    'f[a, b] = sum_{c} x[a, c]*y[c, b]'
    >>> str(dr.einst(f[a, a]))
    'sum_{b, a} x[a, b]*y[b, a]'

which also demonstrates that the tensor definition facility can also be used for
scalar quantities.  :py:class:`TensorDef` is also at the core of the code
optimization and generation facility in the ``gristmill`` package.

Usually for tensorial problems, full simplification requires the utilization of
some symmetries present on the indexed quantities by permutations among their
indices.  For instance, an anti-symmetric matrix entry changes sign when we
transpose the two indices.  Such information can be told to drudge by using the
:py:meth:`Drudge.set_symm` method, by giving generators of the symmetry group
by :py:class:`Perm` instances.  For instance, we can do,

.. testcode::

    dr.set_symm(x, drudge.Perm([1, 0], drudge.NEG))

Then the master simplification algorithm in :py:meth:`Tensor.simplify` is able
to take full advantage of such information.

.. doctest::

    >>> tensor = dr.einst(x[a, b] * v[a] * v[b] + x[b, a] * v[a] * v[b])
    >>> str(tensor)
    'sum_{a, b} x[a, b] * v[a] * v[b]\n + sum_{a, b} x[b, a] * v[a] * v[b]'
    >>> str(tensor.simplify())
    '0'

Normally, drudge subclasses for specific problems add symmetries for some
important indexed bases in the problem.  And some drudge subclasses have helper
methods for the setting of such symmetries, like
:py:meth:`FockDrudge.set_n_body_base` and :py:meth:`FockDrudge.set_dbbar_base`.

For the simplification of the noncommutative vector parts, the base
:py:class:`Drudge` class does **not** consider any commutation rules among the
vectors.  It works on the free algebra, while the subclasses could have the
specific commutation rules added for the algebraic system.  For instance,
:py:class:`WickDrudge` add abstract commutation rules where all the commutators
have scalar values.  Based on it, its special subclass :py:class:`FockDrudge`
implements the canonical commutation relations for bosons and the canonical
anti-commutation relations for fermions.  Also based on it, the subclass
:py:class:`CliffordDrudge` is capable of treating all kinds of Clifford
algebras, like geometric algebra, Pauli matrices, Dirac matrices, and Majorana
fermion operators.  For algebraic systems where the commutator is not always a
scalar, the abstract base class :py:class:`GenQuadDrudge` can be used for
basically all kinds of commutation rules.  For instance, its subclass
:py:class:`SU2LatticeDrudge` can be used for :math:`\mathfrak{su}(2)` algebra in
Cartan-Weyl form.

These drudge subclasses only has the mathematical commutation rules implemented,
for convenience in solving problems, many drudge subclasses are built-in with a
lot of domain-specific information like the ranges and dummies, which are listed
in :ref:`problem_drudges`.  For instance, we can easily see the commutativity of
two particle-hole excitation operators by using the :py:class:`PartHoleDrudge`.


.. doctest::

    >>> phdr = drudge.PartHoleDrudge(spark_ctx)
    >>> t = sympy.IndexedBase('t')
    >>> u = sympy.IndexedBase('u')
    >>> p = phdr.names
    >>> a, i = p.a, p.i
    >>> excit1 = phdr.einst(t[a, i] * p.c_dag[a] * p.c_[i])
    >>> excit2 = phdr.einst(u[a, i] * p.c_dag[a] * p.c_[i])
    >>> comm = excit1 | excit2
    >>> str(comm)
    'sum_{i, a, j, b} t[a, i]*u[b, j] * c[CR, a] * c[AN, i] * c[CR, b] * c[AN, j]\n + sum_{i, a, j, b} -t[a, i]*u[b, j] * c[CR, b] * c[AN, j] * c[CR, a] * c[AN, i]'
    >>> str(comm.simplify())
    '0'

Note that here basically all things related to the problem, like the vector for
creation and annihilation operator, the conventional dummies :math:`a` and
:math:`i` for particle and hole labels, are directly read from the name archive
of the drudge.  Problem-specific drudges are supposed to give such convenience.

In addition to providing context-dependent information for general tensor
operations, drudge subclasses could also provide additional operations on
tensors created from them.  For instance, for the above commutator, we can
directly compute the expectation value with respect to the Fermi vacuum by

.. doctest::

    >>> str(comm.eval_fermi_vev())
    '0'

These additional operations are called tensor methods and are documented in the
drudge subclasses.


Examples on real-world applications
-----------------------------------

In this tutorial, some simple examples are run directly inside a Python
interpreter.  Actually drudge is designed to work well inside Jupyter
notebooks.  By calling the :py:meth:`Tensor.display` method, tensor objects can
be mathematically displayed in Jupyter sessions.  An example of interactive
usage of drudge, we have a `sample notebook`_ in ``docs/examples/ccsd.ipynb``
in the project source.  Also included is a `general script`_ ``gencc.py`` for
the automatic derivation of coupled-cluster theories, mostly to demonstrate
using drudge programmatically.  And we also have a `script for RCCSD theory`_
to demonstrate its usage in large-scale spin-explicit coupled-cluster theories.


.. _sample notebook: https://github.com/tschijnmo/drudge/blob/master/docs/examples/ccsd.ipynb

.. _general script: https://github.com/tschijnmo/drudge/blob/master/docs/examples/gencc.py

.. _script for RCCSD theory: https://github.com/tschijnmo/drudge/blob/master/docs/examples/rccsd.py
