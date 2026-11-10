"""Compare twin prompts that differ by one token, layer by layer.

Given the per-layer hidden states of two nearly identical prompts,
STATE-MIRROR reports the mean cosine and the L2 distance per layer,
the first layer where the states diverge, and the layer with the
largest change. Pure NumPy.
"""

import numpy as np


def _check(states, name):
    if not isinstance(states, (list, tuple)) or len(states) == 0:
        raise ValueError(name + " must be a non-empty list of (T, D) arrays")
    arrs = [np.asarray(s, dtype=float) for s in states]
    for a in arrs:
        if a.ndim != 2:
            raise ValueError(name + " entries must be 2-D (T, D)")
    return arrs


def layer_cosines(states_a, states_b):
    """Mean token-position cosine between twin states, per layer."""
    a = _check(states_a, "states_a")
    b = _check(states_b, "states_b")
    if len(a) != len(b):
        raise ValueError("states_a and states_b must have the same number of layers")
    out = []
    for sa, sb in zip(a, b):
        if sa.shape != sb.shape:
            raise ValueError("twin states must be aligned (same T and D)")
        num = np.sum(sa * sb, axis=1)
        den = np.linalg.norm(sa, axis=1) * np.linalg.norm(sb, axis=1)
        cos = num / np.clip(den, 1e-12, None)
        out.append(float(np.mean(cos)))
    return out


def mirror_report(states_a, states_b, threshold=0.9):
    """Full divergence report for a pair of twin prompts.

    ``threshold`` is the cosine level below which a layer counts as
    diverged. ``first_divergence_layer`` is None when the twins never
    diverge.
    """
    a = _check(states_a, "states_a")
    b = _check(states_b, "states_b")
    cos = layer_cosines(a, b)
    l2 = [float(np.linalg.norm(sa - sb)) for sa, sb in zip(a, b)]
    first = None
    for i, c in enumerate(cos):
        if c < threshold:
            first = i
            break
    return {
        "per_layer_cosine": cos,
        "per_layer_l2": l2,
        "first_divergence_layer": first,
        "max_delta_layer": int(np.argmax(l2)),
    }