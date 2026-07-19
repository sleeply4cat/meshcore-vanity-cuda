# meshcore-vanity (CUDA)

GPU brute-forcer for **MeshCore** vanity public keys. Give it a hex prefix and
it searches Ed25519 keypairs until the raw public key starts with that prefix,
then prints the matching key.

It's an independent, from-scratch CUDA implementation (not a fork). It takes
inspiration from the OpenCL [`nano-vanity`](https://github.com/PlasmaPower/nano-vanity)
project for the MeshCore key derivation and matcher; the low-level Ed25519 field
arithmetic derives from ed25519-donna (public domain). Compared to a naive
"one scalar multiplication per candidate" search it is orders of magnitude
faster (see [How it works](#how-it-works)).

## MeshCore key format

- **Private key** = 64 bytes: `[0..31]` a pre-clamped Ed25519 scalar, `[32..63]`
  an independent random signing component (does **not** affect the public key).
- **Public key** = `clamp(scalar) · B`, compressed to 32 bytes. No Blake2b, no
  base32 — the prefix is matched directly against the raw public-key bytes.
- clamp: `s[0] &= 0xF8; s[31] &= 0x7F; s[31] |= 0x40`.

## Build

```bash
make                      # build for your GPU (default sm_86)
make ARCH=sm_75           # build for a specific architecture
make release              # portable fat binary (sm_60…sm_90 + PTX fallback)
```

Requires the CUDA toolkit (tested with 12.0+) and an NVIDIA GPU. The batch
window is a **runtime** option (see below), so one binary runs on any GPU — no
per-machine rebuild. `make release` produces a single binary that runs across
GPU generations.

## Usage

```
./meshcore-vanity <HEX_PREFIX> [options]
  -l, --limit N     stop after N matches (0 = infinite) [1]
  -w, --window N    batch size/thread: 64|128|256|512|1024 [auto-fit VRAM]
                    bigger = faster but more GPU memory
      --blocks N    CUDA blocks [512]
      --tpb N       threads per block [256]
  -d, --device I    CUDA device index [0]
      --no-progress suppress progress output
      --selftest    run correctness self-tests and exit
```

Output is a 64-hex public key and a 128-hex private key (32-byte scalar +
32-byte random signing component), matching the MeshCore format.

```
$ ./meshcore-vanity abcd
Found matching key!
Public Key:  ABCDBD6497F58B95...82F0B9FD
Private Key: E8F0193C0793...FCAB3C31
```

Each hex nibble is 4 bits, so an N-nibble prefix takes ~2^(4N) attempts on
average; the tool prints `Estimated attempts` at startup. One prefix per run —
launch separate instances for separate prefixes.

> The in-kernel filter matches the low 255 bits of the public key (the `y`
> coordinate). Bit 255 (the `x` parity) is not used for filtering, so the
> effective maximum prefix is 63 hex nibbles — irrelevant in practice. The
> displayed public key is always the full, correct compressed key (recomputed
> and re-verified on a hit).

More operational questions (old distros, CUDA errors, out-of-memory, timing)
are answered in [FAQ.md](FAQ.md).

## How it works

The naive approach computes a full fixed-base scalar multiplication
(`clamp(s)·B`, ~256 point operations) **for every candidate**. This
implementation avoids almost all of that.

### Incremental point addition

Candidates differ by a constant step of 8 in the scalar (the low 3 bits are
cleared by clamp, so `+8` keeps the scalar clamped), and therefore:

```
pubkey(s + 8) = (s + 8)·B = s·B + 8·B = pubkey(s) + 8·B
```

So a whole run of candidates costs one full multiply for the window start plus
one point addition per candidate. `D = 8·B` in niels form is read directly from
donna's base-multiples table, so the step constant is free.

### Affine batched-addition walk (only `y`)

The only expensive part left is the field inversion needed to read off the
affine `y` coordinate. Instead of a sequential projective walk, precompute the
affine multiples `i·D` **once** (they are identical for every thread) and
compute each candidate independently as `P0 + i·D`. Because a match only needs
the `y` bytes, use the complete twisted-Edwards (a = −1) addition and keep
**only `y`**:

```
y_i = (x0·x_i + y0·y_i) / (1 − d·x0·y0·x_i·y_i)
```

With `K = d·x0·y0` computed once per thread and `P_i = x_i·y_i` precomputed in
the shared table, each candidate is: `num = x0·x_i + y0·y_i` (2 mults),
`den = 1 − K·P_i` (1 mult), a shared batch inversion (~3 mults amortized), and
`y = num·inv` (1 mult) — the `x` coordinate is never computed. That's ~7 field
multiplies per candidate instead of hundreds of point operations.

### Montgomery batch inversion

All the per-candidate denominators in a window share a **single** field
inversion (Montgomery's trick: a forward pass of prefix products, one inversion,
a backward pass), which is what makes the affine walk cheap. This is the standard
high-throughput layout used by GPU key searchers.

### No repeated work across launches

Thread `g` owns candidates `s = base + (g·W + j)·8`, `j ∈ [0,W)` — disjoint by
construction within a launch. Between launches the host advances the `base`
counter by exactly the span it just covered (`base += T·W·8`, `T` = total
threads), so launches cover contiguous, non-overlapping intervals. The base
starts from a fresh random value each run; there is no re-rolling and thus no
chance of recomputing the same batch.

### The batch window (`--window`)

Bigger windows amortize the single field inversion over more candidates, so
throughput rises with `W` — but each thread's local buffers (three `W`-element
arrays) grow with it, and the driver reserves that local memory for every
resident thread. So a big window needs more VRAM.

- **`--window N`** picks an explicit size (snapped to 64/128/256/512/1024).
- **default (auto)** queries free VRAM and the per-window footprint and selects
  the largest that fits; on out-of-memory it auto-falls back to a smaller size.

Raise it (`--window 1024`) on a GPU with memory headroom for maximum speed;
lower it under memory pressure. To measure throughput on your GPU, run a search
and read the `Mkeys/s` counter after a few seconds of warm-up.

## Correctness

`./meshcore-vanity --selftest` checks, on the GPU:

1. **Incremental identity** — `(s+8)·B == s·B + 8·B` for 512 random scalars
   (validates `scalarmult` + niels addition, which the step-table precompute
   uses).
2. **Known-answer** — prints `scalar·B` for a fixed scalar; it matches
   `crypto_scalarmult_ed25519_base_noclamp` from PyNaCl / libsodium:

   ```
   scalar 0002030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F60
   pubkey CFE058A4A189EE7230E43A1347EA1A7EEF01F3557991A7FD3CEC8915FD290AEC
   ```

The affine `y`-only formula in the search kernel is validated end-to-end: every
hit is independently re-derived on the host via the *projective* `scalarmult`
path and prefix-checked before being printed, so a wrong affine result could
never produce output.

## Layout

| File | Role |
|---|---|
| `src/ed25519.cuh` | Ed25519 field + group arithmetic (radix 2²⁵·⁵), fixed-base comb, pack. Derived from ed25519-donna (public domain). |
| `src/main.cu` | Step-table precompute kernel, the affine batched-addition search kernel, self-tests, and the host CLI/driver. |
| `Makefile` | `nvcc` build; `make release` builds a portable multi-arch fat binary. |

## License

BSD-2-Clause — see [LICENSE](LICENSE). Credits: the MeshCore derivation and
matcher follow [`nano-vanity`](https://github.com/PlasmaPower/nano-vanity)
(BSD-2-Clause); the Ed25519 arithmetic derives from ed25519-donna (public
domain).
