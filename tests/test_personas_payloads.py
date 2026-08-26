from mindvirus.personas import PERSONAS, Persona
from mindvirus.payloads import PAYLOADS, Payload


def test_ten_unique_personas():
    assert len(PERSONAS) == 10
    names = [p.name for p in PERSONAS]
    assert len(set(names)) == 10
    for p in PERSONAS:
        assert isinstance(p, Persona)
        assert len(p.background) > 40


def test_payloads_complete():
    assert set(PAYLOADS) == {"honesty-absolutism", "ingroup-loyalty"}
    for pid, p in PAYLOADS.items():
        assert isinstance(p, Payload)
        assert p.id == pid
        assert p.belief and p.target_axis and p.judge_rubric


def test_target_axes():
    assert PAYLOADS["honesty-absolutism"].target_axis == "Honesty"
    assert PAYLOADS["ingroup-loyalty"].target_axis == "Loyalty"
