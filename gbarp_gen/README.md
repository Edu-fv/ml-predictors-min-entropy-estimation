# gbarp_gen

gbAR(p) process generator in C and Python. Produces correlated binary sequences parameterized by coefficient vector alpha and noise coefficient beta, subject to `sum(|alpha|) + beta = 1`. Positive coefficients copy the lagged bit; negative coefficients flip it.

## Build

```bash
./build.sh        # produces build/gbarp_gen and build/libgbarp.so
./build.sh clean
```

## Usage

CLI:

```bash
./build/gbarp_gen <output_file> <num_bytes> -a <alpha_vector> [--burn-in <bytes>] [--fast]
```

`-a` takes a comma-separated coefficient vector (p inferred from length). `-p` optionally validates vector length. `--burn-in` sets thermalization bytes (default 10000). `--fast` uses xorshift64 instead of `/dev/urandom`.

```bash
./build/gbarp_gen output.bin 1000000 -a 0.3,-0.2,0.1,-0.05
./build/gbarp_gen output.bin 1000000 -a 0.0625,0.0625,0.0625,0.0625 --fast --burn-in 50000
```

Python:

```python
gbAR(alpha, beta, N_bytes, burn_in_bytes=10000, fast=False) -> bytes
```

`alpha`: coefficient vector (list or numpy array). `beta`: noise coefficient (`1 - sum(|alpha|)`). `N_bytes`: output size. `burn_in_bytes`: thermalization bytes discarded before output. `fast`: use numpy PRNG instead of `os.urandom`.

```python
import numpy as np
from gbarp_gen.python import gbAR, constant_alpha

alpha = constant_alpha(p=8, scaling_factor=0.5)
beta = 1 - sum(abs(alpha))
data = gbAR(alpha, beta, N_bytes=100000)

# Negative coefficients flip the copied bit
alpha = np.array([0.2, -0.15, 0.1, -0.05])
beta = 1 - sum(abs(alpha))
data = gbAR(alpha, beta, N_bytes=100000, burn_in_bytes=50000, fast=True)
```

Alpha family constructors (all return a numpy array):

| Function | Parameters | Shape |
|---|---|---|
| `constant_alpha` | `p, scaling_factor, signs=None` | Equal weights, optional sign pattern |
| `point_to_point_alpha` | `p, scaling_factor` | All weight on lag p |
| `exponentially_decreasing_alpha` | `p, scaling_factor, decay_rate=1` | Exponential decay from lag 1 |
| `gaussian_alpha` | `p, scaling_factor, sigma=None, threshold=0` | Bell curve centered at p/2 |
| `random_pos_alpha` | `p, scaling_factor, seed=None` | Random positive weights |
| `alternating_sign_alpha` | `p, scaling_factor` | [+, -, +, -, ...] * sf/p |

C backend (same signature, optional, faster):

```python
from gbarp_gen.python.c_backend import gbAR_c
data = gbAR_c(alpha, beta, N_bytes=100000)
```

## License

MIT
