"""Tests for the basic permutation facility.
"""

import pickle

import pytest

from drudge import Perm


def test_perm_has_basic_functionality():
    """Test the basic facility for Perms.

    Here a simple cyclic permutation of three points is used to test various
    facilities of Perm.
    """

    pre_images = [1, 2, 0]
    perm = Perm(pre_images, 1)

    assert (len(perm) == len(pre_images))
    for i, v in enumerate(pre_images):
        assert (perm[i] == v)

    assert (perm.acc == 1)

    return


def test_perm_pickles():
    """Tests perms can be correctly pickled."""

    pre_images = [1, 2, 0]
    perm = Perm(pre_images, 1)

    new_args = perm.__getnewargs__()
    assert (len(new_args) == 2)
    assert (new_args[0] == pre_images)
    assert (new_args[1] == 1)
    new_perm = Perm(*new_args)

    pickled = pickle.dumps(perm)
    unpickled_perm = pickle.loads(pickled)

    for form in [new_perm, unpickled_perm]:
        # Maybe equality comparison should be added to Perms.
        assert (len(form) == len(perm))
        for i in range(len(perm)):
            assert (form[i] == perm[i])
        assert (form.acc == perm.acc)

    return


def test_perm_reports_error():
    """Tests perm class reports error for invalid inputs."""

    with pytest.raises(ValueError):
        Perm([1, 1, 1])

    with pytest.raises(ValueError):
        Perm([1, 2])

    with pytest.raises(TypeError):
        Perm('aaa')

    with pytest.raises(TypeError):
        Perm([0, 1], 0.5)
