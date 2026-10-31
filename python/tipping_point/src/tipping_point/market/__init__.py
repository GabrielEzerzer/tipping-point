"""Market regime labels from Binance's free USD-M archive (#29).

archive  the archive's files: fetch, SHA-256 against each .CHECKSUM, parse, row checks
bars     5-minute bars: Garman-Klass realised variance, close, volume, taker-buy share, open
         interest and funding
msvol    the regime labeller: a two-state Gaussian HMM on deseasonalised, baseline-relative log
         RV, refitted weekly by deterministic EM; as-of filtered probabilities, hysteresis,
         episodes, events, flickers and null stretches; the labels run in #22's layout

Stdlib, numpy and pyarrow only. Run `python -m tipping_point.market --help`.
"""
