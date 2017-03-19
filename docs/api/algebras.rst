Support of different algebraic systems
--------------------------------------

.. py:currentmodule:: drudge

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
