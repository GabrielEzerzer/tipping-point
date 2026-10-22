"""The JEPA world model on the 2D Ising model (#27): one shared local encoder for every L.

lattice    circular ConvNeXt encoder, convolutional predictor, torus masks, canonicalisation,
           torus symmetries, bit unpacking                                            (Ising)
data       simulator runs -> packed bursts in RAM, run selection by seed, held-out band (Ising)
objective  MSE on LayerNorm'd targets at hidden cells, VICReg terms, schedules         (generic)
monitor    collapse detector with absolute floors, effective rank, normalised error    (generic)
train      the loop: seeding, AMP per device, checkpoints, resume, manifest, GPU hours (generic)
extract    PCA basis, per-frame surprise and pc_k -> the dataset run's `latents` table
probes     ridge probes from frozen frame embeddings, trained beside untrained

Run `python -m tipping_point.jepa --help`. The design is the #27 research pass (2026-10-22) with
the owner's calls taken as recommended: a per-frame encoder, targets in frame c + delta for
delta in 0..7 inside a burst, bursts flipped to positive magnetisation, training only on
dedicated staircase seeds (never the pre-registered 1-5, never ramps or null runs).
"""

import os

# cuBLAS needs this before CUDA starts for torch.use_deterministic_algorithms(True) to hold
# (torch 2.14 docs); setting it at import keeps every JEPA entry point covered.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
