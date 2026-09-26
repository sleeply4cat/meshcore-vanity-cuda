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
make release              # full: machine code for sm_60…sm_120 + PTX fallback
make release-slim         # slim: PTX only, compiled on the user's machine
```

Requires the CUDA toolkit (tested with 12.0+; `make release` needs 12.8+ for
RTX 50 / sm_120) and an NVIDIA GPU. The batch
window is a **runtime** option (see below), so one binary runs on any GPU — no
per-machine rebuild.

Releases come in two variants, both single binaries that run across GPU
generations:

- **full** (`meshcore-vanity-*`, ~150 MB): machine code for every generation
  from Pascal to Blackwell, plus PTX for GPUs newer than that. Starts at once
  on any listed GPU.
- **slim** (`meshcore-vanity-slim-*`, ~11 MB): no machine code, only PTX. The
  NVIDIA driver compiles it for your GPU on the first run — 4–5 minutes and
  ~2.5 GB of RAM on a laptop; the program says so while it waits — and caches
  the result (`~/.nv/ComputeCache`, `%APPDATA%\NVIDIA\ComputeCache`), so
  later runs start at once. Needs a driver at least as new as the CUDA it was
  built with (R575+ for the releases, built with CUDA 12.9). The machine code
  then comes from the compiler inside your driver rather than the one the full
  build was made and measured with: on an RTX 3050 Laptop (driver 595) it ran
  2.3% slower than the full build.

A GPU the full build has no machine code for (a newer generation) gets the
same one-time compile from its PTX.

## Usage

```
./meshcore-vanity [HEX_PREFIX ...] [options]
  A key counts when it meets ANY of the criteria; all are searched in one pass.
  HEX_PREFIX            key starts with these hex digits (several allowed)
      --repeat-nibble N key starts with N+ identical hex digits (AAAA…)
      --repeat-byte N   key starts with N+ identical bytes (ABABAB…)
  -l, --limit N     stop after N matches total (0 = infinite) [1]
  -w, --window N    batch/thread: 64…16384 (see below) [auto-fit VRAM]
                    past ~2048 at most ~2% faster, but much more GPU memory
      --blocks N    CUDA blocks [auto: 32 whole waves for this GPU]
      --tpb N       threads per block, 32…384 [128]
  -d, --device I    CUDA device index [0]
      --no-progress suppress progress output
      --selftest    run correctness self-tests and exit
      --benchmark   sweep window, block size and grid, print the fastest, exit
```

The defaults are tuned to be close to the best on any GPU (`--tpb 128`, a grid
of 32 whole waves, the largest window ≤ 2048 that fits). To find the exact best
for yours, run `--benchmark`. It takes about a minute: it times every window
that fits, then the block size over the two fastest windows, then the number of
waves for the best pair, and prints the options to use:

```
$ ./meshcore-vanity --benchmark
Benchmarking NVIDIA GeForce RTX 3050 Laptop GPU (16 SM), ~1 minute...
window     Mkeys/s   loc/thr    reserve   (tpb 128, 32 waves)
    64       762.7       1KB      36MB
  ...
  1024      1234.2      20KB     486MB
  1536      1242.3      30KB     726MB
  2048      1260.4      40KB     966MB
  3072      1235.7      60KB    1446MB
  4096      1258.4      80KB    1926MB
  6144      1242.7     120KB    2886MB
  8192    OOM/skip     160KB    3846MB
  ...
  window     tpb   blocks     Mkeys/s   (32 waves)
    2048      64     3072      1244.6
    2048     128     1536      1260.4
    2048     256      512      1206.4
    2048     384      512      1235.7
  ...
  window     tpb   waves   blocks     Mkeys/s
    2048     128       8      384      1253.1
    2048     128      16      768      1252.3
    2048     128      32     1536      1260.4
    2048     128      64     3072      1246.4

Fastest: --window 2048 --tpb 128 --blocks 1536  (1260.4 Mkeys/s)
Differences under ~1% are within run-to-run noise.
```

Here the best is exactly the defaults. Why the block size and the grid matter
at all is explained under [Block size and grid](#block-size-and-grid---tpb---blocks).

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
affine multiples `Q_i = i·D` **once** (they are identical for every thread) and
compute each candidate independently as `P0 + Q_i`, sharing one inversion
across the whole window (see *Montgomery batch inversion*). A match only needs
the `y` bytes, so keep **only `y`** — and use the addition law that makes that
cheapest.

Twisted Edwards curves (`a = −1`) have, besides the complete addition law, a
*dedicated* one (Hisil, Wong, Carter, Dawson, [*Twisted Edwards Curves
Revisited*](https://eprint.iacr.org/2008/522.pdf), 2008) whose `y` does not
involve `d` at all:

```
y3 = (x1·y1 − x2·y2) / (x1·y2 − y1·x2)
```

Divide top and bottom by `y0·y_i`, and let each thread keep `w0 = 1/y0`,
`r0 = x0/y0`, `E = x0·y0`, while the table holds `x_i`, `z_i = 1/y_i`,
`t_i = x_i/y_i` and `s_i = t_i²`:

```
y = w0·(E·z_i − x_i) / (r0 − t_i)
```

It is exceptional only when `P0 = ±Q_i` — for a random base, a 2⁻²⁴⁰ event,
and even then it would only make one thread find nothing in one window (every
hit is re-derived on the host, so no wrong key can come out).

### Walking the window in ±i pairs

Negating an Edwards point flips only `x`: `−Q = (−x, y)`, which flips `x_i` and
`t_i` and keeps `z_i` and `s_i`. So one table entry serves **two** candidates if
the thread's window is walked outwards from its centre instead of forwards from
its start:

```
centre + 8i:   y = w0·(E·z_i − x_i) / (r0 − t_i)
centre − 8i:   y = w0·(E·z_i + x_i) / (r0 + t_i)
```

The two denominators share **one** batch-inversion slot, and their product costs
nothing:

```
(r0 − t_i)·(r0 + t_i) = R − s_i        R = r0², s_i from the table
```

`w0` is folded into the single inversion, so splitting the slot's inverse back
apart with one multiply each (`w0/(r0 − t) = (r0 + t)·inv`, and symmetrically)
yields the whole factor `w0/(r0 ∓ t)` at once. Per pair that is **8 multiplies
and no squarings**: 1 forward (prefix product), 2 backward (prefix inverse,
strip), `E·z_i`, 2 splits and the 2 final products — 4 per candidate. The
pairing also halves both the per-thread buffer and the precomputed table, which
is what lets large windows fit in VRAM.

The first version used the complete formula, `y = (x0·x_i + y0·y_i) / (1 −
d·x0·y0·x_i·y_i)`, where the pair's denominators multiply to `1 − C²`: 11
multiplies and 2 squarings per pair. The d-free walk cut the hot loop from 2459
to 1597 instructions per pair and made the search **~35% faster** (RTX 3050
Laptop, window 2048, measured A/B on an idle GPU; together with the filter
below, ~40%).

### Recompute instead of store

The per-window buffer lives in **local memory (off-chip DRAM)**, so anything
stored there is a full store+load stream per thread. Only the batch-inversion
prefix products actually have to be kept: the slot value `R − s_i` is a
subtraction and is simply redone in the backward pass, and numerators and
split factors come from `r0, E` and the L2-cached shared table. That leaves a
single `W/2`-element buffer for a `W`-candidate window.

Additions and subtractions whose result only feeds a multiply stay unreduced:
the multiply accepts limbs with a couple of bits of headroom, so the carry
chain after them is wasted work.

### Checking the prefix without packing the key

Every candidate has to be filtered, so the filter itself is on the hot path. The
encoding is little-endian, so the prefix lives in the **low** bytes. A field
multiply leaves every limb within its 26/25 bits except limb 1, which may carry
a few bits over, so the value it returns is below `2²⁵⁵ + 2⁵²` and its low 8
bytes are just `f0 + f1·2²⁶ + f2·2⁵¹`: no canonical reduction, no packing. That
value is the canonical `y` unless it lies in `[p, 2²⁵⁵ + 2⁵²)`, i.e. unless
`y < 2⁵² + 19` — a 2⁻²⁰³ chance per candidate, in which case a match would be
missed, never a false one reported. (Doing the full `contract` reduction first
costs ~4% of throughput.)

### Never forming the last product

Each candidate's `y` is the last multiply of its pair, `y = n·d`, and it is used
for nothing but the filter — so it is never formed. Write the product's ten
columns (before any carry) as `m0…m9`, at bit offsets `0, 26, 51, 77, …, 230`.
The multiply's carry chain turns them into `S mod 2²⁵⁵` plus `19·Q` folded into
limb 0, `Q = ⌊S / 2²⁵⁵⌋`, so

```
low 64 bits of y  = (m0 + m1·2²⁶ + m2·2⁵¹ + 19·Q)  mod 2⁶⁴
low 26 bits of y  = (m0 + 19·Q)                    mod 2²⁶
```

and `Q` is fixed by the top columns: the ones below `m7` move `S / 2²⁵⁵` by less
than 2⁻³⁹, so `m7…m9` give it exactly unless the value sits within 2⁻³⁸ of an
integer, and `m8, m9` alone give it unless it sits within 2⁻¹² of one. In those
flagged cases `Q` may be one larger, and both `y` and `y + 19` are tested (the
host re-derives every hit, so an extra candidate costs only a check).

The filter runs in two stages on that basis:

1. **26 bits from 30 products** — columns `m0, m8, m9`, 30 of the 100 partial
   products. Any criterion of 7+ hex digits rejects all but ~2⁻²⁴ of the
   candidates here (for a prefix, a list or a repeat rule alike: each filter
   also has a 26-bit form).
2. **64 bits from 60 products** — `m0, m1, m2, m7, m8, m9`, for the rare
   survivor, then the usual full test.

Stage 2 alone (the first version of this) was worth ~6–8%; stage 1 in front of
it another ~7%; together **+14%** on an RTX 3050 (1099 → 1259 Mkeys/s),
+12–14% with prefix lists and repeat rules. `make EXTRA=-DVANITY_FULL_FINAL_MUL`
builds the old full final multiply for A/B runs.

Things that did **not** help, measured on the same GPU: moving the ×19/×38/×2
constant multiplies from IMAD to shifts and adds on the ALU pipe (−2.5% — the
kernel is bound by instruction issue and latency more than by the IMAD pipe
itself, so shifting work between pipes loses if it adds instructions), keeping
pre-scaled copies of table operands (±0), capping registers at 128 for 16 warps
per SM instead of 12 (−11%, spills), unrolling the backward loop by two pairs
(−3%), and — measured earlier — a 2³²-radix multiply with carry chains (the
`mad.cc` chain runs 2.3× slower per product than `IMAD.WIDE`).

The prefix itself is packed host-side into one `req`/`mask` pair of 64-bit
words, so the test is a single `(y ^ req) & mask` instead of a loop over a
dynamically indexed byte array — which nvcc lowers to a select chain. Prefixes
longer than 8 bytes are checked in full on the host, for the 1-in-2⁶⁴ candidates
that pass the first 8.

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

Advancing in affine coordinates, and keeping `w0 = 1/y0` for the next window,
needs three denominators — which would normally mean another inversion. Instead
the step is **seeded into the existing Montgomery chain** — and placed first, so
its prefix product is the empty product and needs no storage. With the same
d-free formulas:

```
x' = (x0·y0 + xD·yD) / (y0·yD − x0·xD)      = (E + PD) / Dx
y' = (x0·y0 − xD·yD) / (x0·yD − y0·xD)      =  Ny / Dy
w' = 1/y'                                   =  Dy / Ny
```

Once the backward pass has stripped every pair, the accumulator holds
`w0/(Dx·Dy·Ny)`; one multiply by `y0` makes it the plain inverse, and each
denominator's inverse is the product of the other two times that. Preparing a
window therefore costs ~25 multiplies instead of ~600, and no extra live
register. The effect (measured with the earlier complete-formula walk) is largest
where the fixed cost used to dominate:

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
amortises that over more candidates, but it is only ~180 multiplies against ~4
per candidate, so past `W=2048` the whole remaining upside is about **2%**.

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
  2048**, since anything beyond that trades a bounded ~2% for several times the
  memory; on out-of-memory it auto-falls back to a smaller one.

If you want the last percent, use **`--benchmark`** (see *Usage*).

### Block size and grid (`--tpb`, `--blocks`)

The kernel is pinned by `__launch_bounds__` to 168 registers per thread, so an
SM holds 12 warps: a block may use 65536 registers, allocated per warp in units
of 256. That makes the block size matter in steps. `--tpb 256` is the bad one —
two 8-warp blocks do not fit, so an SM runs a single block, 8 warps out of 12 —
and it was the old default (−7% on an RTX 3050, −10% on an RTX 5060 Ti). 384 is
the ceiling (one 12-warp block); larger values are clamped with a message.

Every thread does identical work, so blocks finish in lockstep **waves** of
`SM count × blocks per SM`, and a grid that is not a whole number of waves
leaves most SMs idle during the last one. The old fixed 512 blocks was 3.01
waves on a 170-SM RTX 5090 — four waves of time for three of work — and cost
~5% on an RTX 5060 Ti (36 SM). The grid is now sized with the occupancy
calculator to **32 whole waves** on any GPU.

Many waves matter as well. With several small blocks per SM (the default
`--tpb 128`: three), blocks drift out of phase over the waves, so one block's
serial stretch — the inversion, a long chain of squarings — overlaps other
blocks' multiply-heavy loops instead of every warp on the SM stalling on it at
once. On an RTX 5060 Ti, going from 5 to 40 waves at `--tpb 128` is worth
**~14%**; with one 384-thread block per SM, which cannot drift, the number of
waves makes no difference and it tops out ~12% lower. Together, the new
defaults run the RTX 5060 Ti at ~5.9 Gkeys/s instead of 4.46.

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

3. **Fast filter vs. reference packing** — on multiply outputs, the filter's
   low 64 bits must agree with donna's full 32-byte `contract` for products of
   random field elements and of the edge cases (`0`, `1`, `p−1`, `p`, `p+1`,
   `2²⁵⁵−1`); the only allowed difference is the documented `y < 2⁵²` miss,
   which the edge cases trigger on purpose. Both partial final products (26 and
   64 bits) must agree with the full multiply there too, and on 16.8 million
   random products shaped like the kernel's (unreduced sums and differences
   times multiply outputs), where the ambiguous 26-bit case comes up thousands
   of times.
4. **Ground truth** — every candidate of a small grid is recomputed
   independently (full fixed-base scalar multiplication, projective, packed by
   donna), and the set whose key has a given hex digit must be exactly what the
   kernel records, for a cold launch and for the next one off the carried
   state. This is the test that pins the addition formula itself: the other
   kernel tests compare the kernel with itself, which a formula that was
   consistently wrong would pass.
5. **Prefix list** — searching a list of mixed-length prefixes finds exactly
   the union of what the single-prefix kernel finds for each entry on its own:
   nothing missed, nothing invented. The lengths are deliberately mixed, so a
   filter that ignored the per-entry mask would fail.
6. **Persistent walk** — the search kernel is run with every thread's base
   moved on by one window from a cold seed, and separately at the original bases
   and then once more off the state it carried over;
   with a real prefix the recorded hits depend on the actual points, so the two
   must agree exactly. Nothing else would catch a drifting walk: the recorded
   units stay in range whether or not the points are right.
7. **Window coverage** — the real search kernel is run with an empty prefix, so
   every candidate reports itself and the recorded set is the exact set of
   scalars the window walked. It must be precisely `[0, threads·W)`: no gap, no
   duplicate, no overrun into the neighbouring thread's span. This is what
   guards the ±i walk, whose failure mode is silently losing or repeating
   candidates rather than producing wrong keys. Checked for a power-of-two
   window, a "half" window, and more than one thread.
8. **Repeat rules** — `--repeat-nibble` alone, and both repeat rules together
   with a prefix list, must record exactly what the equivalent explicit prefix
   list records (16 and 274 entries). The nibble rule uses an odd length, where
   the digits are not a contiguous run of bits.
9. **Hit probability** — for several mixes of nested, overlapping and redundant
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
| `Makefile` | `nvcc` build; `make release` / `make release-slim` build the two portable release variants. |

## License

BSD-2-Clause — see [LICENSE](LICENSE). Credits: the MeshCore derivation and
matcher follow [`nano-vanity`](https://github.com/PlasmaPower/nano-vanity)
(BSD-2-Clause); the Ed25519 arithmetic derives from ed25519-donna (public
domain).
