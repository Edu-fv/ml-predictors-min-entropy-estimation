# Copyright (c) 2026 Javier Blanco-Romero
"""ctypes wrapper for libgbarp.so (optional C backend)."""

import ctypes
import os
import numpy as np

_lib = None
_SEARCH_DIRS = [
    os.path.join(os.path.dirname(__file__), "..", "build"),
    os.path.join(os.path.dirname(__file__), ".."),
]


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    for d in _SEARCH_DIRS:
        path = os.path.join(d, "libgbarp.so")
        if os.path.isfile(path):
            _lib = ctypes.CDLL(path)
            _lib.gbarp_generate.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            _lib.gbarp_generate.restype = ctypes.c_int
            return _lib
    return None


def is_available():
    return _load_lib() is not None


def gbAR_c(alpha, beta, N_bytes, burn_in_bytes=10**4, fast=False):
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("C backend not found. Run ./build.sh")

    alpha = np.asarray(alpha, dtype=np.float64)
    c_alpha = alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    output = (ctypes.c_uint8 * N_bytes)()

    ret = lib.gbarp_generate(
        c_alpha, len(alpha), beta, output, N_bytes,
        burn_in_bytes, 0 if fast else 1
    )
    if ret != 0:
        raise RuntimeError(f"gbarp_generate returned {ret}")
    return bytes(output)
