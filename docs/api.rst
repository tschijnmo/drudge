Drudge API reference guide
==========================


Base drudge system
------------------


The base drudge system handles the part of program logic universally applicable
to any tensor and noncommutative algebra system.

.. py:currentmodule:: drudge


Building blocks of the basic drudge data structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Range
    :members:
    :special-members:

.. autoclass:: Vec
    :members:
    :special-members:

.. autoclass:: Term
    :members:
    :special-members:


Canonicalization of indexed quantities with symmetry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some actions are supported to accompany the permutation of indices to indexed
quantities.  All of these accompanied action can be composed by using the
bitwise or operator ``|``.

.. data:: IDENT

    The identitiy action.  Nothing is performed for the permutation.

.. data:: NEG

    Negation.  When the given permutation is performed, the indexed quantity
    needs to be negated.  For instance, in anti-symmetric matrix.

.. data:: CONJ

    Conjugation.  When the given permutation is performed, the indexed quantity
    needs to be taken it complex conjugate.  Note that this action can only be
    used in the symmetry of scalar indexed quantities.

.. autoclass:: Perm
    :members:

.. autoclass:: Group
    :members:


Primary interface
~~~~~~~~~~~~~~~~~

.. autoclass:: Drudge
    :members:
    :special-members:

.. autoclass:: Tensor
    :members:
    :special-members:


Miscellaneous utilities
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sum_

.. autofunction:: prod_

.. autoclass:: TimeStamper
    :members:
    :special-members:


Support of different algebraic systems
--------------------------------------

The base system does not assume any commutation rules amongst the generators of
the algebra, *ie* free algebra or tensor algebra is assumed.  However, by
subclassing the :py:class:`Drudge` class, domain specific knowledge about the
algebraic system in the problem can be given.  Inside drudge, we have some
algebraic systems that is already built in.

.. autoclass:: WickDrudge
    :members:
    :special-members:

.. autoclass:: FockDrudge
    :members:
    :special-members:

.. data:: CR

    The label for creation operators.

.. data:: AN

    The label for annihilation operators.

.. data:: FERMI

    The label for fermion exchange symmetry.

.. data:: BOSE

    The label for boson exchange symmetry.


Direct support of different problems
------------------------------------

In addition to the algebraic rules, more domain specific knowledge can be added
to drudge subclasses for the convenience of working on specific problems.  In
these :py:class:`Drudge` subclasses, we have not only the general mathematical
knowledge like commutation rules, but more detailed information about the
problem as well, like some commonly used ranges, dummies.


.. autoclass:: GenMBDrudge
    :members:
    :special-members:

.. autoclass:: PartHoleDrudge
    :members:
    :special-members:

.. data:: UP

    The symbol for spin up.

.. data:: DOWN

    The symbolic value for spin down.

.. autoclass:: SpinOneHalfGenDrudge
    :members:
    :special-members:

.. autoclass:: SpinOneHalfPartHoleDrudge
    :members:
    :special-members:

