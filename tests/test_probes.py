import pytest

from mindvirus.probes import Battery, ProbeItem, hand_battery, CHOICES, STATEMENT


def test_choices_and_statement():
    assert CHOICES == ["1", "2", "3", "4", "5", "6", "7"]
    assert "acceptable" in STATEMENT


@pytest.mark.parametrize("pid", ["honesty-absolutism", "ingroup-loyalty"])
def test_hand_battery_shape(pid):
    b = hand_battery(pid)
    assert b.payload_id == pid and b.source == "hand"
    assert len(b.items) == 6
    assert sum(i.on_target for i in b.items) == 3
    assert len({i.id for i in b.items}) == 6
    for i in b.items:
        assert i.expected_direction in (-1, 1)
        assert len(i.situation) > 20


def test_hand_battery_unknown_payload():
    with pytest.raises(KeyError):
        hand_battery("nope")


def test_battery_roundtrip(tmp_path):
    b = hand_battery("honesty-absolutism")
    p = tmp_path / "battery.json"
    b.save(p)
    b2 = Battery.load(p)
    assert b2.payload_id == b.payload_id and b2.source == b.source
    assert b2.items == b.items
    assert isinstance(b2.items[0], ProbeItem)
