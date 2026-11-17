# The JEPA world model on the 2D Ising model (#27): one shared local encoder for every L.

import os

# cuBLAS needs this before CUDA starts for torch.use_deterministic_algorithms(True) to hold
# (torch 2.14 docs); setting it at import keeps every JEPA entry point covered.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
