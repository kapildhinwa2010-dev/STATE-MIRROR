# STATE-MIRROR

[![Stars](https://img.shields.io/github/stars/kapildhinwa2010-dev/STATE-MIRROR?style=flat-square)](https://github.com/kapildhinwa2010-dev/STATE-MIRROR/stargazers) [![Forks](https://img.shields.io/github/forks/kapildhinwa2010-dev/STATE-MIRROR?style=flat-square)](https://github.com/kapildhinwa2010-dev/STATE-MIRROR/network/members) [![Issues](https://img.shields.io/github/issues/kapildhinwa2010-dev/STATE-MIRROR?style=flat-square)](https://github.com/kapildhinwa2010-dev/STATE-MIRROR/issues) [![Top language](https://img.shields.io/github/languages/top/kapildhinwa2010-dev/STATE-MIRROR?style=flat-square)](https://github.com/kapildhinwa2010-dev/STATE-MIRROR)

Watch where two nearly identical prompts start to differ inside a model.

STATE-MIRROR takes the per-layer hidden states of twin prompts (same text, one token changed) and reports the mean token cosine and the L2 distance per layer, the first layer where the states diverge beyond a threshold, and the layer with the largest change.

## What it does

- Per-layer mean cosine between aligned twin states.
- Per-layer L2 distance and the largest-change layer.
- Configurable divergence threshold with a first-divergence layer.
- Works on any (T, D) hidden states, from any framework.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from state_mirror.mirror import mirror_report

# states_a / states_b: list of (T, D) arrays, one per layer
states_a = [np.load(f) for f in ["run_a_l0.npy", "run_a_l1.npy", "run_a_l2.npy"]]
states_b = [np.load(f) for f in ["run_b_l0.npy", "run_b_l1.npy", "run_b_l2.npy"]]
report = mirror_report(states_a, states_b, threshold=0.9)
print(report["first_divergence_layer"], report["max_delta_layer"])
```

## How it works

For every layer, the cosine between the corresponding token states of the two prompts is averaged over positions. The first layer whose mean cosine falls below the threshold is where the two runs meaningfully separate.

## Honest scope

- Twin states must be aligned: same number of tokens and the same width.
- A cosine drop measures representational change, not importance or causality.
- Tests use hand-built synthetic states, so no model download is needed.

## License

MIT