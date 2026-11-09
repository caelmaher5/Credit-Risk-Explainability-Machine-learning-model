from src.costs import expected_cost


def test_expected_cost():
    # one bad called good (5) + one good called bad (1) over 4 customers
    assert expected_cost([1, 0, 1, 0], [0, 1, 1, 0]) == 1.5