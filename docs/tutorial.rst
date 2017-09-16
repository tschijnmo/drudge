Drudge tutorial for beginners
=============================

.. currentmodule:: drudge
.. highlight:: Python


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


.. _drs intro:

Drudge scripts
--------------

For maximum flexibility, drudge has been designed to be a Python library from
the beginning.  However, in a lot of cases, like for small tasks or for users
unfamiliar with the Python language or the Spark environment, a domain-specific
language capable of making simple tasks simple can be desired.  Drudge script is
such a language for this purpose.

A drudge script is essentially a Python script heavily tweaked to be executed
inside a special environment.  So all Python lexicographical and syntactical
rules apply.  For a technical description of the pre-processing and execution
drudge scripts, please see :py:meth:`Drudge.exec_drs`.  To execute a drudge
script, we first need a :py:class:`Drudge` object, such that the domain specific
information about the current problem can be available.  For this, we can either
have a normal Python script, where a Drudge object is created with its
:py:meth:`Drudge.exec_drs` called with the source code for the drudge script,
and execute it normally as Python scripts.  Or drudge can also be used as the
main program, either by ``python3 -m drudge`` or ``drudge``.  Then two files
needs to be given as arguments.  The first one is a configuration script, which
is a normal Python script with a Drudge object assigned to a special variable
``DRUDGE``.  Then this Drudge object will be used for the execution of the
actual drudge script given in the second argument.

As an example illustrating the basic principles and ease of drudge scripts, we
assume that we are working on a drudge with a single range registered in the
name archive as ``R``.  To create a symbolic definition of a matrix as a product
of two matrices, suppose the drudge object can be accessed by a variable ``dr``,
we need to write something like::

    p = dr.names
    r = sympy.IndexedBase('r')
    x = sympy.IndexedBase('x')
    y = sympy.IndexedBase('y')
    i, j, k = sympy.symbols('i j k')
    def_ = dr.define(r, (i, p.R), (j, p.R), dr.sum((k, p.R), x[i, k] * y[k, j]))

which can be quite cumbersome for such a simple task.  Suppose the drudge has a
resolver capable of resolving any index to the range, we can write::

    r = sympy.IndexedBase('r')
    x = sympy.IndexedBase('x')
    y = sympy.IndexedBase('y')
    i, j, k = sympy.symbols('i j k')
    def_ = dr.define_einst(r[i, j], x[i, k] * y[k, j])

which although is simplified a lot, still contains quite a lot of noise. Because
of the Python execution model and scoping rules, the indexed bases and symbols
must be explicitly created before they can be used.

Inside a drudge script, names in the name archive, all methods of the current
drudge object, as well as names from the drudge, gristmill (if installed), and
the SymPy package can directly be used without any qualification. More
importantly, Symbol objects and IndexedBase objects are no longer needed to be
explicitly created.  All undefined names will be resolved as an atomic symbol,
which can be construed as both a SymPy symbol and a SymPy IndexedBase.  With
these, the above definition can be simplified into::

    def_ = define_einst(r[i, j], x[i, k] * y[k, j])

Due to the ubiquity of tensor definitions in common drudge tasks, a special
operator ``<<=`` (Python left-shift augmented assignment operator) is introduced
for the making definitions.  With this, the above definition can be written as::

    r[i, j] <<= sum((k, R), x[i, k] * y[k, j])

which makes the definition and put the definition in the name archive by
:py:meth:`Drudge.set_name`.  So by default, the definition is put into the name
archive under name ``r`` as a :py:class:`TensorDef` object, and the base of the
definition is put under name ``_r``.  Since names in the name archive do not
need to be qualified in drudge scripts::

    sum((k, R), r[i, k] * r[k, j])

directly gives us the chain product :math:`\mathbf{XYXY}`.  And symbolic
references to the ``r`` tensor without the concrete definition substituted in
can still be made by using ``_r``, like::

    s = sum((k, R), _r[i, k] * _r[k, j])

which gives us the product :math:`\mathbf{RR}`.  For this, the actual definition
can be substituted explicitly when desired, for example, by::

    s.subst(r)

which gives us :math:`\mathbf{XYXY}`.

Note that the definition by ``<<=`` is made by using the :py:meth:`Drudge.def_`
method.  As a result, when the drudge property :py:meth:`Drudge.default_einst`
is set, Einstein summation convention is going to be automatically applied to
the right-hand side.  So we can simply write::

    r[i, j] <<= x[i, k] * y[k, j]

when the ranges of :math:`i, j, k` can be resolved by the drudge.

In cases where tainting of the global name archive is undesired for a tensor
definition, we can use the ``<=`` operator, which simply returns the definition
object without adding it to the name archive.  For instance, to store the tensor
definition in a variable ``def_``, we can use::

    def_ = r[i, j] <= x[i, k] * y[k, j]

This can be useful in functions inside drudge scripts.

Additionally, drudges could have more functions specifically to be used inside
drudge scripts.  For instance, in the base :py:class:`Drudge` class, we have a
simple constructor ``S``, for converting strings to the special kind of symbols
that can be indexed and used in ``<<=`` in drudge scripts.  Also have ``sum_``
for the actual Python built-in ``sum`` function, which is shadowed by the
:py:meth:`Drudge.sum` method.  And the drudge object used for the execution can
be accessed by ``DRUDGE``.

For the taste of users without much object-oriented programming, inside drudge
scripts, method calling like ``obj.meth(args)`` can also be written as
``meth(obj, args)``.  For instance, for a tensor ``tensor``::

    simplify(tensor)

is equivalent to::

    tensor.simplify()

Attribute access can be done in the same way, for instance,::

    n_terms(tensor)

is equivalent to::

    tensor.n_terms

Note that a caveat of this syntactic sugar is that the method name cannot be
defined to be anything else before the calling.  For instance,::

    n_terms = 10
    n_terms(tensor)

does not work, since ``n_terms`` is already defined to the integer 10, thus
cannot be called any more.  Another caveat is that static methods cannot be
called in this way, which fortunately does not appear a lot in common usages of
drudge.

For the convenience of symbolic computation, all integer literals inside drudge
scripts are automatically resolved to SymPy integer values, rather than the
built-in integer values.  As a result, we can directly write::

    1 / 2

for the rational value of one-half, without having to worry about the truncation
or degradation to finite-precision floating-point numbers for Python integers.
To access built-in integers, which is normally unnecessary, we can explicitly
write something like ``int(1)``.

For convenience of users, some drudge functions has got slightly different
behaviour inside drudge scripts.  For instance, the :py:meth:`Tensor.simplify`
method will eagerly compute the result and repartition the terms among the
workers.  And tensors also have more readable string representation inside
drudge scripts.


Examples on real-world applications
-----------------------------------

In Python interface
~~~~~~~~~~~~~~~~~~~

In this tutorial, some simple examples are run directly inside a Python
interpreter.  Actually drudge is designed to work inside Jupyter notebooks as
well.  By calling the :py:meth:`Tensor.display` method, tensor objects can be
mathematically displayed in Jupyter sessions.  An example of interactive usage
of drudge, we have a `sample notebook`_ in ``docs/examples/ccsd.ipynb`` in the
project source.  Also included is a `general script`_ ``gencc.py`` for the
automatic derivation of coupled-cluster theories, mostly to demonstrate using
drudge programmatically.  And we also have a `script for RCCSD theory`_ to
demonstrate its usage in large-scale spin-explicit coupled-cluster theories.


.. _sample notebook: https://github.com/tschijnmo/drudge/blob/master/docs/examples/ccsd.ipynb

.. _general script: https://github.com/tschijnmo/drudge/blob/master/docs/examples/gencc.py

.. _script for RCCSD theory: https://github.com/tschijnmo/drudge/blob/master/docs/examples/rccsd.py


As drudge scripts
~~~~~~~~~~~~~~~~~

For drudge scripts, we have two example scripts both deriving the classical CCD
theory.  Both of them is based on the following configuration script
``conf_ph.py``,

.. literalinclude:: examples/conf_ph.py

Here we only set a simple :py:class:`PartHoleDrudge` without much modification.
To illustrate the most basic usage of drudge scripts, we have example
``ccd.drs``,

.. literalinclude:: examples/ccd.drs
    :language: Python

With the comment described in the above script, we can see that drudge script
can bare a lot of resemblance to the mathematical notation.  To make a
derivation of the many-body theory, we basically just use the operators like
``+``, ``*``, and ``|`` to do arithmetic operations on the tensors and use
``simplify`` to get the result simplified.

For another more advanced example, we have the ``ccd_adv.drs`` script,

.. literalinclude:: examples/ccd_adv.drs
    :language: Python

In the example ``ccd.drs``, it is attempted to be emphasized that drudge scripts
are very similar to common mathematical notation and should be easy to get
started.  In this ``ccd_adv.drs`` example, the power and flexibility of drudge
scripts being actually Python scripts is emphasized.  Foremost, rather than
spelling each order of commutation out, here the similarity-transformed
Hamiltonian :math:`\bar{\mathbf{H}}` is computed by using a Python loop.  This
can be helpful for repetitive tasks.  Also the computation of
:math:`\bar{\mathbf{H}}` is put inside a function.  Being able to define and
execute functions makes it easy to reuse code inside drudge scripts. Here, the
function is given to the :py:meth:`Drudge.memoize` function.  So its result is
automatically dumped into the given pickle file.  When the file is already
there, the result will be directly read and used with the execution of the
function skipped.  This can be helpful for large multi-step jobs.

Note that ``<<=`` is used to make the working equations as tensor definitions of
class :py:class:`TensorDef`.  In drudge scripts,::

    variable = tensor

assigns the tensor ``tensor`` to the variable ``variable``.  The variable is a
normal Python variable and works in the normal Python way.  And the tensor is
just a static expression of its mathematical content, with all the free symbols
being free.  At the same time,::

    lhs <<= tensor

defines the ``lhs`` as the tensor, with the definition pushed into the name
archive of the drudge.  By using :py:class:`TensorDef` objects, we also have a
left-hand side, which enables the accompanying `gristmill`_ package to optimize
the evaluation of the entire array by its advanced algorithms.

.. _gristmill: https://github.com/tschijnmo/gristmill

For the result, here they are written into a very structured LaTeX output, which
can be easily compiled into PDF files.  Note that by using the
:py:meth:`Report.add` function with different arguments, we can create
structured report with sections and descriptions for the equations.


Common caveats
--------------

When using drudge, there are some common pitfalls that might confuse
beginning users, here we attempt a small summary of the prominent ones for
convenience.  Note that users are encouraged to go through SymPy tutorial
first, where some common caveats about using SymPy is summarized.


.. rubric:: Importing drudge

In this tutorial, ``import drudge`` and ``import sympy`` is used and we need to
give fully-qualified name to refer to objects in them.  Normally, it can be
convenient to use ``from drudge import *`` to import everything from drudge. For
these cases, it needs to be careful that the importation of all objects from
drudge needs to follow the importation of all objects from SymPy, or the SymPy
``Range`` class will shallow the actual class for symbolic range in drudge.


.. rubric:: Wrong names in drudge scripts

In drudge scripts, all unresolved names evaluates to symbols with the given
name, similar to the behaviour of dedicated computer algebra systems like
Mathematica or Maple.  In this way, extra care need to be taken for names inside
drudge scripts.  Although drudge attempts to give as sensible error message as
possible, sometimes quite confusing errors can be given for a wrongly typed
name.  For this cases, running drudge scripts inside a debugger can be helpful.
Whenever the error comes from an object that is an ``DrsSymbol`` instance, it is
highly-likely there is a typo in the drudge script at this place.  Inside the
debugger, ``up`` command can be used to move the stack to the place in the
drudge script, then the trouble-maker can be attempted to be identified.


.. rubric:: Name clashing, symbol names and variable names

In Python scripts, normally we would bind atomic symbols to variables named
the same as the symbol itself, like::

    i = Symbol('i')

Sometimes we would later accidentally bind the variable with something else,
this could stop the symbol being accessible by its name any more.  For
example, after ::

    for i in range(10):
        print(i)

``i`` can no longer be used for the symbol :math:`i`.  This can give some highly
obscure bugs.  Similarly, in drudge scripts, whenever a variable is created with
a given name, symbols with the same name cannot be access by simply spelling the
name out any more.  For example, with the above ``for`` loop, symbol :math:`i`,
might not be accessed by ``i`` any more, which is actually bound to integer
number 9 now.

To resolve these issues, generally the variable name can be mangled some how,
for instance, by appending an underscore in the variable name.

Very similarly, sometimes obscure error could occur when symbolic objects like
indexed bases and symbols are named in the same way.  For instance, when an
indexed base is named in the same way as an index, it could be changed when the
index is substituted.  So it is highly recommended that indexed bases and
symbols are free from any name clashing.  For cases with clashing, the indexed
base names could be mangled with something like trailing underscore to avoid
problems.
