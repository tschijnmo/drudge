"""Project-wide shared test fixtures."""

import pytest
from pyspark import SparkConf, SparkContext


@pytest.fixture(scope='session', autouse=True)
def spark_ctx():
    """A simple spark context."""

    conf = SparkConf().setMaster('local[2]').setAppName('drudge-unittest')
    ctx = SparkContext(conf=conf)

    return ctx
