"""Market regime labels from Binance's free USD-M archive (#29).

archive  the archive's files: fetch, SHA-256 against each .CHECKSUM, parse, row checks
bars     5-minute bars: Garman-Klass realised variance, close, volume, taker-buy share, open
         interest and funding
msvol    the regime labeller: a two-state Gaussian HMM on deseasonalised, baseline-relative log
         RV, refitted weekly by deterministic EM; as-of filtered probabilities, hysteresis,
         episodes, events, flickers and null stretches; the labels run in #22's layout
cascade  the liquidation-cascade proxy: open interest falling while price and taker flow move
         together on high volume, merged into asset instances and market events; the pooled list
         as a labels run's table `cascades`
nulls    the GARCH(1,1)-t simulator, the detection-delay simulation, the event-time shift null and
         the circularity floor, which takes its scorers and AUC as inputs
reports  the label-count and stability reports and the floor's tables, each as Parquet plus a
         Markdown summary for the issue

Stdlib, numpy and pyarrow only. Run `python -m tipping_point.market --help`. The floor runs from
harness/market_floor.py, since no module here may import the feature code.
"""
