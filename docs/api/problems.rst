.. _problem_drudges:

Direct support of different problems
------------------------------------

.. py:currentmodule:: drudge

In addition to the algebraic rules, more domain specific knowledge can be added
to drudge subclasses for the convenience of working on specific problems.  In
these :py:class:`Drudge` subclasses, we have not only the general mathematical
knowledge like commutation rules, but more detailed information about the
problem as well, like some commonly used ranges, dummies.


General problems for many-body theories
+++++++++++++++++++++++++++++++++++++++

.. autoclass:: GenMBDrudge
    :members:
    :special-members:


Problems for many-body theories based the particle-hold picture
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. autoclass:: PartHoleDrudge
    :members:
    :special-members:


Many-body theories with explicit spin
+++++++++++++++++++++++++++++++++++++

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

.. autoclass:: RestrictedPartHoleDrudge
    :members:
    :special-members:


Other many-body theories
++++++++++++++++++++++++

.. autoclass:: BogoliubovDrudge
    :members:
    :special-members:
