"""Property tests on the autocorrelation estimator: the FFT rho(t) of each replica, averaged over
the replicas, equals the direct lagged sum of products over the zero-lag sum."""

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from tipping_point.ising import autocorr


@given(
    arrays(
        np.float64,
        st.tuples(st.integers(1, 3), st.integers(2, 60)),
        elements=st.integers(-3, 3).map(float) | st.floats(-100.0, 100.0),
    )
)
def test_autocorrelation_is_the_direct_lagged_sum_over_the_zero_lag_sum(x):
    rows = x[x.std(axis=1) > 0]
    assume(len(rows) > 0)
    y = rows - rows.mean(axis=1, keepdims=True)
    n = y.shape[1]
    sums = np.array([[np.dot(r[: n - t], r[t:]) for t in range(n)] for r in y])
    want = (sums / sums[:, :1]).mean(axis=0)
    np.testing.assert_allclose(autocorr.autocorrelation(x), want, atol=1e-9)
