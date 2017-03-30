Introduction
============

Drudge is a computer algebra system based on SymPy for noncommutative and tensor
algebras, with a specific emphasis on many-body theory.  To get started, Python
of version at least 3.5 and a C++ compiler with good C++14 support are needed.
In the development of drudge, Clang++ 3.9 and g++ 6.3 has been fully tested.

Also Apache Spark of version later than 2.1 is needed for the parallel execution
of drudge.  For small tasks without requirement on parallelization, a fork of
the DummyRDD_ project can be used in place of an actual Spark context.  For
parallel execution on supercomputers managed by the SLURM queueing system, the
script in spark-in-slurm_ can be helpful.  Throughout the entire documentation,
``SparkContext()`` will be used to create a minimal Spark context for
demonstration purpose.  The actual context should be created according to your
site.

As an experimental project, the documentation can be outdated, incomplete,
incorrect, or have a lot of bad formatting.  For any confusion, UTSL.


.. _DummyRDD: https://github.com/tschijnmo/DummyRDD
.. _spark-in-slurm: https://github.com/tschijnmo/spark-in-slurm

