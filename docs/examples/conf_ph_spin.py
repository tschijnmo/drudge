"""Configuration for particle-hole problem with explicit spin.
"""

from pyspark import SparkContext
from drudge import RestrictedPartHoleDrudge

ctx = SparkContext()
dr = RestrictedPartHoleDrudge(ctx)
dr.full_simplify = False

DRUDGE = dr
