# gbAR(p) generator
# Copyright (c) 2026 Javier Blanco-Romero

"""
gbAR(p) generator, Python implementation.

X_t = (Σ a+_i X_{t-i} + Σ a-_i (1 - X_{t-i}) + b e_t) mod 2

All randomness (categorical + noise) is pre-read before the loop.
Secure (default): os.urandom. Fast: numpy PRNG seeded from os.urandom.
"""

import os
import numpy as np
from collections import deque
from bitarray import bitarray


def _generate_random_bytes(num_bytes, fast=False, rng=None):
    if fast:
        if rng is None:
            rng = np.random.default_rng(
                int.from_bytes(os.urandom(8), 'big')
            )
        return rng.bytes(num_bytes), rng
    return os.urandom(num_bytes), None


# Alpha family constructors

def point_to_point_alpha(p, scaling_factor):
    alpha = np.zeros(p)
    alpha[p - 1] = 1
    return scaling_factor * alpha


def constant_alpha(p, scaling_factor, signs=None):
    if signs is None:
        alpha = np.ones(p) / p
    else:
        alpha = np.array([x / len(signs) for x in signs])
    return scaling_factor * alpha


def exponentially_decreasing_alpha(p, scaling_factor, decay_rate=1):
    alpha = np.array([np.exp(-decay_rate * i) for i in range(p)])
    alpha /= np.sum(alpha)
    return scaling_factor * alpha


def gaussian_alpha(p, scaling_factor, sigma=None, threshold=0):
    if sigma is None:
        sigma = max(p / 4, 1)
    x = np.arange(p)
    mu = (p - 1) / 2
    alpha = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    alpha[alpha < threshold] = 0
    if np.sum(alpha) > 0:
        alpha /= np.sum(alpha)
    else:
        alpha = np.ones(p) / p
    return scaling_factor * alpha


def random_pos_alpha(p, scaling_factor, seed=None):
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0, 1, size=p)
    alpha /= np.sum(alpha)
    return scaling_factor * alpha


def alternating_sign_alpha(p, scaling_factor):
    signs = np.array([(-1) ** j for j in range(p)])
    return signs * scaling_factor / p


# Core generator

def gbAR(alpha, beta, N_bytes, burn_in_bytes=10**4, fast=False):
    alpha = np.asarray(alpha)
    p = len(alpha)
    burn_in_bits = 8 * burn_in_bytes
    N_bits = N_bytes * 8 + burn_in_bits

    P_abs = np.abs(np.concatenate([alpha, [beta]]))
    P_abs /= P_abs.sum()
    P_cumsum = np.cumsum(P_abs)
    alpha_pos = alpha >= 0

    cat_bytes = N_bits * 4
    noise_bytes = N_bytes + burn_in_bytes

    rng = None
    if fast:
        rng = np.random.default_rng(int.from_bytes(os.urandom(8), 'big'))
        cat_raw, rng = _generate_random_bytes(cat_bytes, fast=True, rng=rng)
        noise_raw, rng = _generate_random_bytes(noise_bytes, fast=True, rng=rng)
    else:
        cat_raw, _ = _generate_random_bytes(cat_bytes)
        noise_raw, _ = _generate_random_bytes(noise_bytes)

    cat_vals = np.frombuffer(cat_raw, dtype='>u4').astype(np.float64) / 0xFFFFFFFF
    cat_idx = np.clip(np.searchsorted(P_cumsum, cat_vals, side='left'), 0, len(P_abs) - 1)
    et = np.unpackbits(np.frombuffer(noise_raw, dtype=np.uint8))

    window = deque([0] * p, maxlen=p)
    out = bitarray()
    noise_cat = len(P_abs) - 1

    for t in range(N_bits):
        k = cat_idx[t]
        if k == noise_cat:
            X_t = et[t]
        else:
            h = window[-(k + 1)]
            X_t = h if alpha_pos[k] else (1 - h)

        window.append(X_t)
        if t >= burn_in_bits:
            out.append(X_t)

    return out.tobytes()
