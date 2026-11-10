import numpy as np
import pytest

from state_mirror.mirror import layer_cosines, mirror_report

T, D = 3, 4
BASE = np.ones((T, D))
ALT = np.zeros((T, D))
ALT[:, 1] = 1.0
OTHER = np.zeros((T, D))
OTHER[:, 2] = 1.0


def test_layer_cosines_parallel_and_identical():
    cos = layer_cosines([BASE, ALT, ALT * 2.0], [BASE, ALT, ALT * 5.0])
    assert len(cos) == 3
    assert cos[0] == pytest.approx(1.0)
    assert cos[1] == pytest.approx(1.0)
    assert cos[2] == pytest.approx(1.0)


def test_divergence_detected():
    a = [BASE.copy(), BASE.copy(), ALT.copy(), ALT.copy()]
    b = [BASE.copy(), BASE.copy(), OTHER.copy(), OTHER.copy()]
    r = mirror_report(a, b, threshold=0.9)
    assert r["per_layer_cosine"][0] == pytest.approx(1.0)
    assert r["per_layer_cosine"][1] == pytest.approx(1.0)
    assert r["per_layer_cosine"][2] == pytest.approx(0.0)
    assert r["first_divergence_layer"] == 2
    assert r["max_delta_layer"] == 2
    assert r["per_layer_l2"][0] == pytest.approx(0.0)
    assert r["per_layer_l2"][2] > 0.0


def test_no_divergence():
    a = [BASE.copy(), ALT.copy(), ALT.copy()]
    r = mirror_report(a, [x.copy() for x in a])
    assert r["first_divergence_layer"] is None
    assert all(c == pytest.approx(1.0) for c in r["per_layer_cosine"])
    assert all(d == pytest.approx(0.0) for d in r["per_layer_l2"])


def test_threshold_sensitivity():
    a = [BASE.copy(), BASE.copy(), BASE.copy()]
    b = [BASE.copy(), ALT.copy(), BASE.copy()]
    r_strict = mirror_report(a, b, threshold=1.0)
    assert r_strict["first_divergence_layer"] == 1
    r_loose = mirror_report(a, b, threshold=0.0)
    assert r_loose["first_divergence_layer"] is None


def test_errors():
    a = [BASE.copy(), ALT.copy()]
    with pytest.raises(ValueError):
        layer_cosines([], a)
    with pytest.raises(ValueError):
        layer_cosines(a, a[:1])
    with pytest.raises(ValueError):
        layer_cosines([np.zeros((2, 2, 2))] * 2, [np.zeros((2, 2, 2))] * 2)
    with pytest.raises(ValueError):
        layer_cosines(a, [np.zeros((5, D)) for _ in a])