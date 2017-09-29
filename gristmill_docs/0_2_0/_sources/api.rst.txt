API Reference
-------------

The ``gristmill`` package can be divided into two orthogonal parts,

The evaluation optimization part,
    which transforms tensor definitions into a mathematically equivalent
    definition sequence with less floating-point operations required.

The code generation part,
    which takes tensor definitions, either optimized or not, into computer code
    snippets.


.. py:currentmodule:: gristmill


Evaluation Optimization
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: optimize

.. autofunction:: verify_eval_seq

.. autofunction:: get_flop_cost


Code generation
~~~~~~~~~~~~~~~

.. autoclass:: BasePrinter
    :members:
    :special-members:

.. autoclass:: ImperativeCodePrinter
    :members:
    :special-members:

.. autoclass:: CCodePrinter
    :members:
    :special-members:

.. autoclass:: FortranPrinter
    :members:
    :special-members:

.. autoclass:: EinsumPrinter
    :members:
    :special-members:
