"""Tests for the permutation group class."""

import pickle

from drudge import Perm, Group


def test_s3_group_correct_and_serializable():
    """Test the S3 group.

    Since the permutation group objects are mostly opaque, here it is tested
    by its new arguments, which is also further verified to work with pickle.

    """

    cycle = Perm([1, 2, 0])
    transp = Perm([1, 0, 2], 1)
    group = Group([cycle, transp])

    args = group.__getnewargs__()
    assert len(args) == 1
    transvs = args[0]

    assert len(transvs) == 2
    top_level = transvs[0]
    lower_level = transvs[1]

    assert len(top_level) == 2
    target = top_level[0]
    transv = top_level[1]
    assert target == 0
    assert len(transv) == 2
    assert set(i[0][0] for i in transv) == {1, 2}

    assert len(lower_level) == 2
    target = lower_level[0]
    transv = lower_level[1]
    assert target == 1
    assert len(transv) == 1
    perm = transv[0]
    # This permutation fixes 0, but is not identity.  It must be the
    # transposition of 1 and 2
    assert perm[0][0] == 0
    assert perm[0][1] == 2
    assert perm[0][2] == 1
    assert perm[1] == 1

    # Args should not change from the group reconstructed from pickle.
    pickled = pickle.dumps(group)
    new_group = pickle.loads(pickled)
    assert new_group.__getnewargs__() == args

    # Assert that Group type should be considered true, since it implements
    # neither __bool__ nor __len__.
    assert group
