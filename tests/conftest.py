"""Project-wide shared test fixtures."""

import os

import pytest

IF_DUMMY_SPARK = 'DUMMY_SPARK' in os.environ

@pytest.fixture(scope='session', autouse=True)
def spark_ctx():
    """A simple spark context."""

    if IF_DUMMY_SPARK:
        from dummy_spark import SparkConf, SparkContext
        conf = SparkConf()
        ctx = SparkContext(master='', conf=conf)
    else:
        from pyspark import SparkConf, SparkContext
        conf = SparkConf().setMaster('local[2]').setAppName('drudge-unittest')
        ctx = SparkContext(conf=conf)

    return ctx


def skip_in_spark(**kwargs):
    """Skip the test in Apache Spark environment.

    Mostly due to issues with pickling some SymPy objects, some tests have to
    be temporarily skipped in Apache Spark environment.
    """
    return pytest.mark.skipif(not IF_DUMMY_SPARK, **kwargs)
