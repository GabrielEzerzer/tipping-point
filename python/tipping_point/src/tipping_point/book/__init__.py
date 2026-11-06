"""Order-book features for the market JEPA (#30 M2).

features  one day's `book_samples` (the engine's 100 ms grid, 20 levels a side) reduced to a
          1-second feature table on the day's UTC grid, 86,400 rows with a gap mask

Run `python -m tipping_point.book <samples.parquet>`, which is what `python -m
tipping_point.market book --reduce "python -m tipping_point.book"` calls per day. The labeller
(`tipping_point.market`) imports nothing from here, and nothing here imports the labeller's
`msvol` or `cascade` (tests/market/test_circularity.py).
"""
