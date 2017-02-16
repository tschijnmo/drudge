"""Test for the name inject utility."""

import types

from drudge import Drudge


def test_drudge_injects_names():
    """Test the name injection method of drudge."""

    # Dummy drudge.
    dr = Drudge(types.SimpleNamespace(defaultParallelism=1))

    string_name = 'string_name'
    dr.set_name(string_name)
    dr.set_name(1, 'one')

    dr.inject_names(suffix='_')

    assert string_name_ == string_name
    assert one_ == 1
