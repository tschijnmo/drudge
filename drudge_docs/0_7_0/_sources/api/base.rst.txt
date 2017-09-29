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

.. autoclass:: TensorDef
    :members:
    :special-members:


Miscellaneous utilities
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sum_

.. autofunction:: prod_

.. autoclass:: Stopwatch
    :members:
    :special-members:

.. autoclass:: Report
    :members:
    :special-members:


