"""Property tests on the labeller's run bounds and hysteresis: the bounds of each run of True
rebuild the mask they came from, and the hysteresis equals the bar-by-bar state machine."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from tipping_point.market import msvol


@given(arrays(np.bool_, st.integers(0, 40)))
def test_run_bounds_rebuild_their_mask(mask):
    starts, ends = msvol.runs(mask)
    rebuilt = np.zeros(len(mask), dtype=bool)
    for start, end in zip(starts, ends, strict=True):
        assert start < end
        rebuilt[start:end] = True
    assert (rebuilt == mask).all()
    assert (starts[1:] > ends[:-1]).all()


LEVELS = st.integers(0, 8).map(lambda k: k / 8) | st.floats(0.0, 1.0)


@given(
    arrays(np.float64, st.integers(0, 40), elements=LEVELS | st.just(np.nan)),
    LEVELS,
    LEVELS,
)
def test_hysteresis_is_the_bar_by_bar_state_machine(p, enter, exit):
    state, want = 0, []
    for value in p:
        if np.isnan(value):
            want.append(-1)
            continue
        if value <= exit:
            state = 0
        elif value >= enter:
            state = 1
        want.append(state)
    assert msvol.hysteresis(p, enter, exit).tolist() == want
