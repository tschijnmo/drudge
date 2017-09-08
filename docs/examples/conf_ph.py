"""Configures a simple drudge for particle-hole model."""

from dummy_spark import SparkContext
from drudge import PartHoleDrudge

ctx = SparkContext()
dr = PartHoleDrudge(ctx)
dr.full_simplify = False

DRUDGE = dr
