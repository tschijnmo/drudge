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

.. autoclass:: Perm
    :members:
    :special-members:

.. autoclass:: Group
    :members:
    :special-members:


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


In addition to the algebraic rules, more domain specific knowledge can be added
to drudge subclasses for the convenience of working on specific problems.

.. autoclass:: GenMBDrudge
    :members:
    :special-members:

.. autoclass:: PartHoleDrudge
    :members:
    :special-members:

.. autoclass:: SpinOneHalfGenDrudge
    :members:
    :special-members:

.. autoclass:: SpinOneHalfPartHoleDrudge
    :members:
    :special-members:

