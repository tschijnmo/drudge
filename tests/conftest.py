"""Project-wide shared test fixtures."""

import os

import pytest


@pytest.fixture(scope='session', autouse=True)
def spark_ctx():
    """A simple spark context."""

    if 'DUMMY_SPARK' in os.environ:
        from dummy_spark import SparkConf, SparkContext
        conf = SparkConf()
        ctx = SparkContext(master='', conf=conf)
    else:
        from pyspark import SparkConf, SparkContext
        conf = SparkConf().setMaster('local[2]').setAppName('drudge-unittest')
        ctx = SparkContext(conf=conf)

    return ctx
