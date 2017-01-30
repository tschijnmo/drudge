"""Tests for the canonicalization facility for Eldags."""

from drudge import Perm, Group
from drudge.canonpy import canon_eldag


def test_eldag_can_be_canonicalized():
    """Tests the Eldag canonicalization facility.

    Note that this test more focuses on better coverage in the canonpy interface
    to libcanon, rather than on the correctness of canonicalization algorithm,
    which should be already tested within libcanon.

    In this test, we have two bivalent nodes in the Eldag, one without symmetry,
    one with symmetry.  They are both connected to two terminal nodes with the
    same colour.

    In this graph, the connection to the non-symmetric node determines the
    resulted permutations.
    """

    transp = Perm([1, 0], 1)
    symms = [None, Group([transp]), None, None]
    colours = [0, 1, 1, 1]  # We force the non-symmetric node to come earlier.

    for if_same in [True, False]:
        # If the non-symmetric node is connected to the two terminal nodes in
        # order.  The symmetric node always connect to them in order.

        edges = [2, 3, 2, 3] if if_same else [3, 2, 2, 3]
        ia = [0, 2, 4, 4, 4]

        node_order, perms = canon_eldag(edges, ia, symms, colours)

        # Assertions applicable to both cases.
        assert node_order[0] == 0
        assert node_order[1] == 1
        for i in [0, 2, 3]:
            assert perms[i] is None
            continue

        # The ordering of the two terminals.
        if if_same:
            assert node_order[2:] == [2, 3]
        else:
            assert node_order[2:] == [3, 2]

        # The permutation of the symmetric node.
        perm = perms[1]
        if if_same:
            assert perm[0] == 0
            assert perm[1] == 1
            assert perm.acc == 0
        else:
            assert perm[0] == 1
            assert perm[1] == 0
            assert perm.acc == 1

        continue

    return
