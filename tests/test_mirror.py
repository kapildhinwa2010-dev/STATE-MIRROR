import numpy as np
import pytest

from state_mirror.mirror import layer_cosines, mirror_report

T, D = 3, 4


def make_twins():
    """Twin prompts identical in layers 0-1, orthogonal from layer 2 on."""
    base = np.ones((T, D))
    alt = np.zeros((T, D))
    alt[:, 1] = 1.0
    states_a = [base.copy(), base.copy(), alt.copy(), alt.copy() * 2.0]
    states_b = [base.copy(), base.copy(), alt.copy(), alt.copy() * 5.0]
    return states_a, states_b


def test_layer_cosines_identical_then_orthogonal():
    a, b = make_twins()
    cos = layer_cosines(a, b)
    assert len(cos) == 4
    assert cos[0] == pytest.approx(1.0)
    assert cos[1] == pytest.approx(1.0)
    # layer 2: identical arrays, cosine 1
    assert cos[2] == pytest.approx(1.0)
    # layer 3: parallel vectors scaled differently, cosine 1
    assert cos[3] == pytest.approx(1.0)


def test_divergence_detected():
    a, _ = make_twins()
    diverge = np.zeros((T, D))
    diverge[:, 2] = 1.0
    b = [a[0].copy(), a[1].copy(), diverge, diverge]
    r = mirror_report(a, b, threshold=0.9)
    assert r["per_layer_cosine"][0] == pytest.approx(1.0)
    assert r["per_layer_cosine"][1] == pytest.approx(1.0)
    assert r["per_layer_cosine"][2] < 0.5
    assert r["first_divergence_layer"] == 2
    assert r["max_delta_layer"] == 2


def test_no_divergence():
    a, _ = make_twins()
    r = mirror_report(a, [x.copy() for x in a])
    assert r["first_divergence_layer"] is None
    assert all(c == pytest.approx(1.0) for c in r["per_layer_cosine"])
    assert all(d == pytest.approx(0.0) for d in r["per_layer_l2"])


def test_threshold_sensitivity():
    a, _ = make_twins()
    b = [a[0].copy(), a[0] * 2.0, a[0], a[0]]
    r_strict = mirror_report(a, b, threshold=1.0)
    assert r_strict["first_divergence_layer"] == 1
    r_loose = mirror_report(a, b, threshold=0.0)
    assert r_loose["first_divergence_layer"] is None


def test_errors():
    a, _ = make_twins()
    with pytest.raises(ValueError):
        layer_cosines([], a)
    with pytest.raises(ValueError):
        layer_cosines(a, a[:2])
    with pytest.raises(ValueError):
        layer_cosines([np.zeros((2, 2, 2))] * 2, [np.zeros((2, 2, 2))] * 2)
    with pytest.raises(ValueError):
        layer_cosines(a, [np.zeros((5, D)) for _ in a])