Drudge
------

Drudge is a symbolic algebra system built on top of the `SymPy`_ library with
its primary focus on tensorial and non-commutative algebras.  It is motivated by
complex symbolic problems in quantum chemistry and many-body theory, but it
could be useful for any tedious and error-prone symbolic manipulation and
simplification in any problem with indexed quantities, symbolic summations, and
non-commutative algebra.

Built on the generic algorithm for the canonicalization of combinatorial
objects, like strings and graphs, in `libcanon`_, drudge is able to find a
canonical form for mathematical expressions with tensors with symmetries and
symbolic summations.  For instance, for a 4th-order tensor *u* with symmetry

.. image:: https://latex.codecogs.com/svg.latex?u_%7Babcd%7D%3D-u_%7Bbacd%7D%3D-u_%7Babdc%7D%3Du_%7Bbadc%7D
    :align: center
    :alt: u_{abcd}=-u_{bacd}=-u_{abdc}=u_{badc}

expression like

.. image:: https://latex.codecogs.com/svg.latex?%5Csum_%7Bcd%7Du_%7Bacbd%7D%5Crho_%7Bdc%7D-%5Csum_%7Bcd%7Du_%7Bcabd%7D%5Crho_%7Bdc%7D%5Csum_%7Bcd%7Du_%7Bcdbc%7D%5Crho_%7Bcd%7D
    :align: center
    :alt: \sum_{cd}u_{acbd}\rho_{dc}-\sum_{cd}u_{cabd}\rho_{dc}+\sum_{cd}u_{cdbc}\rho_{cd}

can be automatically simplified into a single term like,

.. image:: https://latex.codecogs.com/svg.latex?3%5Csum_%7Bcd%7Du_%7Bacbd%7D%5Crho_%7Bdc%7D
    :align: center
    :alt: 3\sum_{cd}u_{acbd}\rho_{dc}

despite the initial different placement of the indices to the symmetric *u*
tensor and different naming of the dummy indices for summations.

In addition to the full consideration of the combinatorial properties of
symmetric tensors and summations during the simplification, drudge also offers a
general system for handling non-commutative algebraic systems.  Currently,
drudge directly supports the `CCR and CAR algebra`_ for treating fermions and
bosons in many-body theory, general `Clifford algebras`_, and `su(2) algebra`_
in its Cartan-Killing basis.  Other non-commutative algebraic systems should be
able to be added with ease.


Drudge is developed by Jinmo Zhao and Prof Gustavo E Scuseria at Rice
University, and was supported as part of the Center for the Computational Design
of Functional Layered Materials, an Energy Frontier Research Center funded by
the U.S. Department of Energy, Office of Science, Basic Energy Sciences under
Award DE-SC0012575.


.. _SymPy: http://www.sympy.org
.. _libcanon: https://github.com/tschijnmo/libcanon
.. _CCR and CAR algebra: https://en.wikipedia.org/wiki/CCR_and_CAR_algebras
.. _Clifford algebras: https://en.wikipedia.org/wiki/Clifford_algebra
.. _su(2) algebra: https://en.m.wikipedia.org/wiki/Special_unitary_group#Lie_Algebra