"""The query-side PLAID device rule: config decides the class, the replica decides the ordinal.

The hazard these tests exist for: the service sets `RAGTIME_QUERY_PLAID_DEVICE` per replica, so a
device resolution that honours the config class but not that variable opens every PLAID replica on
the config's `cuda:0`, stacking 25.3 GiB on card 0 instead of sitting at 12.6/12.6 across two
cards. `resolve_query_plaid_device` is the seam the rule lives on, and these assertions are cheap
enough to run anywhere -- the surrounding path needs a GPU and a real index to reach.
"""

from __future__ import annotations

import pytest

from ragtime.preprocess.index import IndexIntegrityError, resolve_query_plaid_device

pytestmark = pytest.mark.small


def test_a_replicas_ordinal_is_adopted() -> None:
    """Config says cuda:0, replica 1 says cuda:1, and cuda:1 wins.

    The broken version returned "cuda:0" here, which is how two replicas ended up on one card.
    """
    assert resolve_query_plaid_device("cuda:0", "cuda:1") == "cuda:1"
    assert resolve_query_plaid_device("cuda:0", "cuda:5") == "cuda:5"


def test_an_unset_variable_leaves_the_configured_device_untouched() -> None:
    """The common case: no env var, no surprise. `None` and `""` both mean 'unset'."""
    assert resolve_query_plaid_device("cuda:0", None) == "cuda:0"
    assert resolve_query_plaid_device("cuda:0", "") == "cuda:0"
    assert resolve_query_plaid_device("cpu", None) == "cpu"


def test_agreeing_with_the_config_is_a_no_op() -> None:
    assert resolve_query_plaid_device("cuda:0", "cuda:0") == "cuda:0"
    assert resolve_query_plaid_device("cpu", "cpu") == "cpu"


@pytest.mark.parametrize(
    ("configured", "env"),
    [("cuda:0", "cpu"), ("cpu", "cuda:0"), ("cuda:1", "cpu"), ("cpu", "cuda:3")],
)
def test_crossing_the_device_CLASS_still_raises(configured: str, env: str) -> None:
    """Adopting the ordinal does not weaken the guard.

    The device class is shared across a run family and it reorders results: scores were
    byte-identical on 8 of 20 queries and the id order matched on 14 of 20 at depth 100.
    Nothing on disk records which device produced a result, so a silent cuda/cpu switch
    would be unattributable.
    """
    with pytest.raises(IndexIntegrityError, match="different device class"):
        resolve_query_plaid_device(configured, env)


def test_the_two_rules_are_independent() -> None:
    """An ordinal difference is not a class difference, and neither is the other.

    An implementation that returned `env_device` unconditionally would pass every test above
    except the class ones, and one that compared the two strings whole would raise on a
    legitimate replica.
    """
    # ordinal differs, class same -> adopted, no raise
    assert resolve_query_plaid_device("cuda:2", "cuda:0") == "cuda:0"
    # class differs even though both carry an ordinal -> raise
    with pytest.raises(IndexIntegrityError):
        resolve_query_plaid_device("cuda:2", "cpu:0")
