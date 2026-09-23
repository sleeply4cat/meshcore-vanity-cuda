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
  an independent random signing component. It does **not** affect the public
  key, but it keys the nonce of every signature, so it is exactly as secret as
  the scalar: whoever knows it can recover the scalar from one signature.
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
./meshcore-vanity [HEX_PREFIX ...] [options]
  A key counts when it meets ANY of the criteria; all are searched in one pass.
  HEX_PREFIX            key starts with these hex digits (several allowed)
      --repeat-nibble N key starts with N+ identical hex digits (AAAA…)
      --repeat-byte N   key starts with N+ identical bytes (ABABAB…)
  -l, --limit N     stop after N matches total (0 = infinite) [1]
  -w, --window N    batch/thread: 64…16384 (see below) [auto-fit VRAM]
                    past ~2048 at most ~1.3% faster, but much more GPU memory
      --blocks N    CUDA blocks [512]
      --tpb N       threads per block, 32…384 [256]
  -d, --device I    CUDA device index [0]
      --no-progress suppress progress output
      --selftest    run correctness self-tests and exit
      --benchmark   sweep window + block size, print the fastest, and exit
```

Run `--benchmark` first: it sweeps every window that fits your GPU, then sweeps
the block size over the top two windows, and prints the config to use:

```
$ ./meshcore-vanity --benchmark
window     Mkeys/s   loc/thr    reserve
   512        621.9     10KB     246MB
  1024        633.7     20KB     486MB
  2048        635.3     40KB     966MB
  6144        643.5    120KB    2887MB
  8192     OOM/skip    160KB    3846MB
...
Grid sweep (top windows x block size):
  window     tpb     Mkeys/s
    6144     128       661.4
    6144     256       644.1
    6144     384       682.3
Fastest: --window 6144 --tpb 384  (682.3 Mkeys/s)
```

(The top two windows are swept because the per-block register limit can bar a
large `tpb` on the biggest window while a slightly smaller one still allows it.
The block size is not monotone either: resident threads per SM move in steps, so
128 and 384 can both beat 256.)

Output is a 64-hex public key and a 128-hex private key (32-byte scalar +
32-byte random signing component), matching the MeshCore format.

```
$ ./meshcore-vanity abcd
Found matching key!
Public Key:  ABCDBD6497F58B95...82F0B9FD
Private Key: E8F0193C0793...FCAB3C31
```

Each hex nibble is 4 bits, so an N-nibble prefix takes ~2^(4N) attempts on
average; the tool prints `Estimated attempts` at startup.

Several prefixes can be searched in one pass — a hit on any of them counts, so
the expected time divides by their number:

```
$ ./meshcore-vanity beef cafe f00d --limit 1
Searching for keys that match:
  prefix BEEF             req BEEF mask FFFF
  prefix CAFE             req CAFE mask FFFF
  prefix F00D             req F00D mask FFFF
Estimated attempts: 2^14.4
...
Found matching key!
Prefix:      CAFE
Public Key:  CAFE7C1D...
```

Redundant entries are dropped with a note — `cafe` next to `ca` adds nothing,
since every key starting with `CAFE` already starts with `CA`.

This beats running one instance per prefix, which would split the GPU between
them and test one prefix each at a fraction of the throughput. A list tests every
entry against the *same* arithmetic, so it costs only the extra comparisons —
about 1.5% of throughput per entry (3.5% for two, 14% for eight), against an
N-fold reduction in expected time. A single prefix runs a separate kernel and
pays nothing at all for the feature.

### Any "pretty" key: repeat rules

If you don't care *which* digit, only that the key starts with a run of one:

```
$ ./meshcore-vanity --repeat-nibble 7 --repeat-byte 4 --limit 3
Searching for keys that match:
  7+ identical hex digits (0000000, AAAAAAA, ...)
  4+ identical bytes (ABABABAB, ...)
Estimated attempts: 2^23.0
...
Found matching key!
Repeat:      4 identical bytes
Public Key:  03030303F392166F...
Found matching key!
Repeat:      8 identical hex digits
Public Key:  EEEEEEEE3BAB2146...
```

- `--repeat-nibble N`: the first N hex digits are all the same (`AAAAAAA…`).
  Any of 16 digits will do, so it costs as much as a specific prefix one digit
  shorter.
- `--repeat-byte N`: the first N bytes are all the same (`ABABAB…`, and also
  `AAAAAA…`, since `AA` is a byte too). Any of 256 bytes will do, so it costs
  as much as a specific prefix one *byte* shorter.

The printed `Repeat:` line gives the run the key actually has, which can be
longer than asked. Both rules combine with each other and with prefixes, and a
key meeting any of them counts; the estimate accounts for the overlaps exactly
(e.g. `EEEEEEEE` meets both rules above, and is counted once). The check itself
is a masked compare against the first digit or byte broadcast across the word,
as cheap as the single-prefix one, and each combination of criteria is its own
kernel, so a search never pays for a check it does not use.

While looking for the first key, the progress line also reports how likely it is
that a match was already inside the span searched so far — `1 − (1 − p)^attempts`,
where `p` is the per-candidate chance of meeting any of the criteria:

```
Tried 3892314112 keys (282.9 Mkeys/s), 59.6% chance it was already in range
```

That is the honest way to read a long run: the average is not a deadline, and
this number tells you where you actually are on the curve. Passing 50% is
expected roughly at the average; sitting at 95% without a hit is unlucky but not
evidence anything is wrong.

> The in-kernel filter only looks at the first 8 bytes of the key (the low
> bits of `y`); every candidate that passes it is recomputed on the host as the
> full, correct compressed key — `x` parity bit included — and checked against
> the whole criterion, so prefixes and repeat runs can be up to 64 hex digits
> long. Anything past 16 digits is far out of reach anyway.

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

### Walking the window in ±i pairs

Negating an Edwards point flips only `x`: `−Q = (−x, y)`. So one table entry
`i·D` serves **two** candidates at once if the thread's window is walked
outwards from its centre instead of forwards from its start. Writing
`A = x0·x_i`, `B = y0·y_i` and `C = K·P_i` (three mults, shared by the pair):

```
centre + 8i:   y = (B + A) / (1 − C)
centre − 8i:   y = (B − A) / (1 + C)
```

The two denominators then collapse into **one** batch-inversion slot, because

```
(1 − C)·(1 + C) = 1 − C²          — a single squaring
```

and the joint inverse is split back apart with two mults
(`1/(1−C) = (1+C)·inv`, and symmetrically). Per pair that is ~10 mults + 2
squarings — about **5.5 mults per candidate instead of 8** — and it halves both
the per-thread buffer and the precomputed table, which is what lets much larger
windows fit in VRAM.

### Recompute instead of store

The per-window buffer lives in **local memory (off-chip DRAM)**, so anything
stored there is a full store+load stream per thread. Only the batch-inversion
prefix products actually have to be kept: `C`, both denominators and both
numerators are **recomputed** in the backward pass from `x0,y0,K` and the
L2-cached shared table. That leaves a single `W/2`-element buffer for a
`W`-candidate window. Recomputing costs one extra mult + one extra squaring per
pair, and is net *faster* — cheap ALU and cached table reads in exchange for
expensive local-memory traffic.

### Checking the prefix without packing the key

Every candidate has to be filtered, so the filter itself is on the hot path. Two
things make it nearly free. The encoding is little-endian, so the prefix lives in
the **low** bytes: after the same canonical reduction `contract` performs (which
cannot be skipped — radix 2²⁵·⁵ folds the top of the value back into limb 0 via
`2²⁵⁵ ≡ 19`, so the low bits depend on every limb), the low 8 bytes are just
`f0 | f1<<26 | f2<<51`, and the other 24 bytes are never assembled. And the
prefix itself is packed host-side into one `req`/`mask` pair of 64-bit words, so
the test is a single `(y ^ req) & mask` instead of a loop over a dynamically
indexed byte array — which nvcc lowers to a select chain. Together that is worth
~8% of total throughput. Prefixes longer than 8 bytes fall back to the byte
compare, on a path only a 1-in-2⁶⁴ candidate ever reaches.

### Multiply-accumulate in one instruction

The field multiply is ~100 `32×32→64` products accumulated into ten 64-bit
lanes. Written as `m += (uint64_t)x*y` that is three SASS instructions per
product — `IMAD.WIDE.U32` plus an `IADD3`/`IADD3.X` pair for the 64-bit add.
Expressed as inline PTX `mad.lo.cc.u32` + `madc.hi.u32`, ptxas folds it into a
single accumulating `IMAD.WIDE.U32`. The hot loop drops from 2760 to 2324
instructions and throughput rises ~6%. The cost is that every pair passes
through the one carry flag, so independent accumulators cannot be interleaved as
freely — this was measured, not assumed. Build with `-DCURVE25519_NO_PTX_MAC` to
fall back to plain C.

### The walk persists across launches

Setting up a window used to cost a full fixed-base scalar multiplication for the
thread's centre point plus an inversion to make it affine — about 600 field
multiplies per thread per launch, paid again every launch. Instead the thread
**keeps its centre** and advances it by one window, `D = (W·8)·B`, while the host
counts the thread's launches since its base was drawn, so the two stay in
lockstep.

Advancing in affine coordinates needs the two addition denominators `1 ± C`
(`C = K·xD·yD`), which would normally mean another inversion. Instead the
step is **seeded into the existing Montgomery chain** — and placed first, so its
prefix product is the empty product and needs no storage. Once the backward pass
has stripped every pair, the accumulator is left holding exactly `1/(1−C²)`, and
both denominators fall out of it with one multiply each:

```
x' = (x0·yD + y0·xD) / (1 + C)      1/(1+C) = (1 − C)·acc
y' = (y0·yD + x0·xD) / (1 − C)      1/(1−C) = (1 + C)·acc
```

Preparing a window therefore costs ~12 multiplies instead of ~600, and not one
extra live register. The effect is largest where the fixed cost used to dominate:

| window | before | after |
|---|---|---|
| 512 | 522 | 645 Mkeys/s |
| 1024 | 590 | 656 |
| 2048 | 625 | 658 |
| 6144 | 645 | 657 |

which is the real point: throughput no longer depends much on `W`, so the window
can be chosen for its memory footprint instead of its speed.

### Montgomery batch inversion

All the denominators in a window share a **single** field inversion
(Montgomery's trick: a forward pass of prefix products, one inversion, a backward
pass), which is what makes the affine walk cheap. Thanks to the ±i pairing the
batch has only `W/2` slots for `W` candidates. This is the standard
high-throughput layout used by GPU key searchers.

### Randomness, independent keys, no repeated work

Every thread `g` has **its own random base** `base_g`, drawn from the OS CSPRNG
(`getrandom` / `BCryptGenRandom`) and clamped. In its `k`-th launch on that base
it owns candidates `s = base_g + (k·W + j)·8`, `j ∈ [0,W)`: consecutive launches
continue the same contiguous run, and the `+i` half of the ±i walk stops one
short of the next window's first unit, so nothing is visited twice. Runs of
different threads start 2²⁵⁰-ish apart at random, so they never meet in
practice.

The per-thread base matters for the keys, not for speed. A private key is its
base plus a known small offset, so two keys found on **one** base would differ
by a small, easily brute-forced multiple of 8 — leaking one would give away the
other. So a thread yields **at most one key per base**: when it produces a hit,
the host checks and prints it, gives that thread a fresh random base (re-seeded
with one fixed-base multiply on its next launch), and ignores its other hits
from the same window. Keys from different bases are independent, so `--limit N`
gives N unrelated keys. Only very loose criteria, which one thread meets several
times in one window, lose anything to this.

Randomness is drawn only when a base is drawn and for each printed key's signing
half — never on the GPU path — so this costs no throughput.

### The batch window (`--window`)

A window now costs exactly one per-thread field inversion — the shared Montgomery
one (the window-start inversion went away with the persistent walk). A bigger `W`
amortises that over more candidates, but it is only ~152 multiplies against ~5.5
per candidate, so past `W=2048` the whole remaining upside is under **1.3%**.

The cost is GPU memory. Each thread's single buffer holds one prefix product per
±i **pair**, i.e. `W/2` field elements = `W·20` bytes of local memory, and **the
driver reserves that for the SM's full thread capacity, not the kernel's actual
occupancy** — so the real reservation is

```
reserve ≈ SM_count × maxThreadsPerSM × W·20 bytes
```

which grows fast: on a 16-SM / 1536-threads-per-SM GPU, `W=2048` pins ~0.95 GB
and `W=6144` ~2.8 GB. That is why big windows still need substantial VRAM even
though their *live* occupancy is low.

- **`--window N`** picks an explicit size, snapped to a supported one:
  64/128/256/512/1024/2048/4096/8192/16384 plus the ×1.5 "half" sizes
  1536/3072/6144/12288 (all multiples of 512) that fill the gaps so a window can
  be chosen closer to what VRAM allows. 16384 is the ceiling, at ~320 KB/thread
  (under the 512 KB local limit); its ~7.7 GB reserve on a 16-SM GPU fits only on
  8 GB+ cards, so on 4 GB cards the practical top is 4096–6144.
- **default (auto)** selects the largest window that fits free VRAM **capped at
  2048**, since anything beyond that trades a bounded ~1.3% for several times the
  memory; on out-of-memory it auto-falls back to a smaller one.

If you want the last percent, use **`--benchmark`**: it measures `Mkeys/s` and
the memory reserve for every window (unfitting ones show `OOM/skip`), then sweeps
the block size over the top two windows and prints the `--window`/`--tpb` to use.
Do sweep `--tpb`
too: resident-warps-per-SM granularity makes the best block size non-obvious —
128 and 384 can both beat 256. 384 is the ceiling: a block may use 65536
registers, allocated per warp in units of 256, and the kernel is pinned by
`__launch_bounds__` to the 168 registers per thread that let a 12-warp block
fit. Larger `--tpb` is clamped with a message.

## Correctness

`./meshcore-vanity --selftest` checks, on the GPU:

1. **Incremental identity** — `(s+8)·B == s·B + 8·B` for 512 random scalars
   (validates `scalarmult` + niels addition, which the step-table precompute
   uses).
2. **Known-answer** — `scalar·B` for a fixed scalar must equal
   `crypto_scalarmult_ed25519_base_noclamp` from PyNaCl / libsodium (the
   expected value is built in; a mismatch fails the run):

   ```
   scalar 0002030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F60
   pubkey CFE058A4A189EE7230E43A1347EA1A7EEF01F3557991A7FD3CEC8915FD290AEC
   ```

3. **Fast filter vs. reference packing** — the low-64-bit fast path must agree
   with donna's full 32-byte `contract` on 4096 random field elements plus the
   canonicalisation edge cases (`0`, `1`, `p−1`, `p`, `p+1`, `2²⁵⁵−1`).
4. **Prefix list** — searching a list of mixed-length prefixes finds exactly
   the union of what the single-prefix kernel finds for each entry on its own:
   nothing missed, nothing invented. The lengths are deliberately mixed, so a
   filter that ignored the per-entry mask would fail.
5. **Persistent walk** — the search kernel is run with every thread's base
   moved on by one window from a cold seed, and separately at the original bases
   and then once more off the state it carried over;
   with a real prefix the recorded hits depend on the actual points, so the two
   must agree exactly. Nothing else would catch a drifting walk: the recorded
   units stay in range whether or not the points are right.
6. **Window coverage** — the real search kernel is run with an empty prefix, so
   every candidate reports itself and the recorded set is the exact set of
   scalars the window walked. It must be precisely `[0, threads·W)`: no gap, no
   duplicate, no overrun into the neighbouring thread's span. This is what
   guards the ±i walk, whose failure mode is silently losing or repeating
   candidates rather than producing wrong keys. Checked for a power-of-two
   window, a "half" window, and more than one thread.
7. **Repeat rules** — `--repeat-nibble` alone, and both repeat rules together
   with a prefix list, must record exactly what the equivalent explicit prefix
   list records (16 and 274 entries). The nibble rule uses an odd length, where
   the digits are not a contiguous run of bits.
8. **Hit probability** — for several mixes of nested, overlapping and redundant
   prefixes and rules, every 5-digit key is enumerated; the matching fraction
   must equal the computed estimate exactly, and dropping redundant criteria
   must not change which keys match.

The affine `y`-only formula in the search kernel is validated end-to-end too:
every hit is independently re-derived on the host via the *projective*
`scalarmult` path and checked against the full criterion before being printed, so a wrong affine
result could never produce output.

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
