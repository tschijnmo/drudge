---
layout: page
title: Drudge / Gristmill
---

> {{ site.description }}


Based on [SymPy](www.sympy.org), drudge/gristmill aims to perform complex
symbolic manipulations involving tensor and non-commutative algebra, and
optionally generates highly optimized implementations automatically for their
numerical computations.  In spite of its root in quantum many-body theories,
the stack is applicable to any problem with heavy dependency on tensor algebra
and tensor computations.

Drudge is the component focusing on the general symbolic manipulation and
simplification.  Inside its core, the high-performance C++
[libcanon](https://github.com/tschijnmo/libcanon) library is used for the
canonicalization of the DAG for tensor expressions.  Based on drudge, gristmill
could rewrite given tensor computations to perform substantial optimization,
whose result can be used by its simple C/Fortran/Python code generator.


## Drudge


Compare with common general computer algebra systems, drudge simplifies with
full consideration of symmetries of tensors and symbolic summations.  For
instance, for a 4th-order tensor \\(u\\) with symmetry

$$
u_{abcd} = -u_{bacd} = -u_{abdc} = u_{badc}
$$

expression like

$$\sum_{cd} u_{acbd} \rho_{dc} - \sum_{cd} u_{cabd} \rho_{dc} + \sum_{cd} u_{cdbc} \rho_{cd}$$

can be automatically simplified into a single term like,

$$
3 \sum_{cd} u_{acbd} \rho_{dc}
$$

despite the initial different placement of the indices to the symmetric \\(u\\)
tensor and different naming of the dummy indices for summations.

In addition to the full consideration of the combinatorial properties of
symmetric tensors and summations during the simplification, drudge also offers
a general system for handling non-commutative algebraic systems.  Currently,
drudge directly supports the CCR and CAR algebra for treating fermions and
bosons in many-body theory, general Clifford algebras, and
\\(\mathfrak{su}(2)\\) algebra in its Cartan-Killing basis.  Other
non-commutative algebraic systems can be added with ease.

## Gristmill


Based on drudge, gristmill utilizes novel advanced algorithms to efficiently
parenthesize and factorize tensor computations for less arithmetic cost.  For
instance, a matrix chain product

$$
\mathbf{R} = \mathbf{A} \mathbf{B} \mathbf{C}
$$

can be parenthesized into

$$
\mathbf{R} = \left( \mathbf{A} \mathbf{B} \right) \mathbf{C}
$$

or

$$
\mathbf{R} = \mathbf{A} \left( \mathbf{B} \mathbf{C} \right)
$$

depending on which one of them incurs less arithmetic cost for the given shapes
of the matrices.  With just a small overhead relative to specialized dynamic
programming code for matrix chain products, general tensor contractions are
supported.  For instance, the ladder term in the CCD theory in quantum
chemistry

$$
r_{abij} = \sum_{c,d=1}^v \sum_{k,l=1}^o v_{klcd} t_{cdij} t_{abkl}
$$

can be automatically parenthesized into a two-step evaluation

$$
\begin{aligned}
    p_{klij} &= \sum_{c,d=1}^v v_{klcd} t_{cdij}\\
    r_{abij} &= \sum_{k,l=1}^o p_{klij} t_{abkl}\\
\end{aligned}
$$

Because of the efficiency of the algorithm, contraction of even twenty factors
can be handled well.


When computing sums of multiple contractions, factorizations of some or all
terms leading to savings of arithmetic cost can also be automatically found.
For instance, the correlation energy of the CCSD theory in quantum chemistry,

$$
e = \frac{1}{4} \sum_{i,j=1}^o \sum_{a,b=1}^{v} u_{ijab} t^{(2)}_{abij}
+ \frac{1}{2} \sum_{i,j=1}^o \sum_{a,b=1}^v u_{ijab} t^{(1)}_{ai} t^{(1)}_{bj}
$$

can be automatically rewritten into

$$
    e = \frac{1}{4} \sum_{i,j=1}^o \sum_{a,v=1}^v u_{ijab} \left(
        t^{(2)}_{abij} + 2 t^{(1)}_{ai} t^{(1)}_{bj}
    \right)
$$

which takes less arithmetic cost.

In addition to parenthesization and factorization, gristmill also has additional
optimization heuristics like common symmetrization optimization.  The same
intermediates can also be guaranteed to be computed only once by the
canonicalization power of drudge.


The code generator is a component orthogonal to the optimizer.  Both optimized
and unoptimized computation can be given for naive Fortran or C code (with
optional OpenMP parallelization), or Python code using NumPy or TensorFlow
libraries.


## Acknowledgements

Drudge is developed by Jinmo Zhao and Prof Gustavo E Scuseria at Rice
University, and was supported as part of the Center for the Computational Design
of Functional Layered Materials, an Energy Frontier Research Center funded by
the U.S. Department of Energy, Office of Science, Basic Energy Sciences under
Award DE-SC0012575.


<script type="text/javascript"
    src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
