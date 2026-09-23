# FAQ

### What does this do?
It brute-forces MeshCore Ed25519 keypairs on the GPU until the public key starts
with a hex prefix you choose (a "vanity" key). It's a from-scratch CUDA rewrite
of the OpenCL `nano-vanity`/MeshCore fork, ~100× faster on the same GPU.

### Are the keys real MeshCore keys?
Yes. The output is exactly the MeshCore format: a 64-hex public key and a
128-hex private key (`[32-byte clamped scalar][32-byte random signing half]`).
The public key is `clamp(scalar)·B` compressed to 32 bytes — no Blake2b, no
base32. Every hit is independently re-derived and re-checked on the host before
being printed, and the math is cross-validated against libsodium/PyNaCl (see
[README → Correctness](README.md#correctness)).

### Do I need a GPU? Which one?
Yes — it's GPU-only, there is no CPU search path. Any NVIDIA GPU with a recent
driver works. It's tuned for Ampere (sm_86), but the released binary is a fat
binary covering sm_60…sm_90 plus a PTX fallback, so it runs on other generations
too. Build from source with `make ARCH=sm_XX` for your specific card.

### How fast is it?
Hundreds of Mkeys/s on a modern NVIDIA GPU — orders of magnitude faster than a
naive per-candidate search. Exact throughput depends on your GPU and the
`--window` setting; measure it by running a search and reading the `Mkeys/s`
counter after a few seconds of warm-up (a cold start — context init, JIT, boost
ramp — roughly halves the first few seconds).

### How long will my prefix take?
Each hex nibble is 4 bits, so an N-nibble prefix needs ~2^(4N) attempts on
average. As a rough guide at ~700 Mkeys/s: 6 nibbles ≈ instant, 8 nibbles ≈
~6 s, 10 nibbles ≈ ~25 min, 12 nibbles ≈ ~4.5 days. The tool prints
`Estimated attempts: 2^bits` at startup.

Those are averages, not deadlines — the search is memoryless, so being twice
over the average means nothing is wrong. While hunting the first key the
progress line shows the actual figure of merit: the chance a match was already
inside the span searched so far, `1 - (1 - p)^attempts`, where `p` is the
per-candidate chance of meeting any of your criteria. It
crosses 50% around the average and 95% at about three times it.

### Can I match multiple prefixes, a suffix, or a regex?
Multiple prefixes: yes — pass them all and any hit counts, which divides the
expected time by their number:

```
./meshcore-vanity beef cafe f00d
```

Running one instance per prefix instead would be strictly worse: the instances
split the GPU, so each tests one prefix at a fraction of the throughput, while a
list tests all of them against the same arithmetic.

The list costs about 1.5% of throughput per entry — 3.5% for two prefixes, 14%
for eight — so the trade stays lopsided in your favour well past a dozen. A
single prefix uses a separate kernel and pays nothing for the feature.

Suffixes and regexes are not supported. A suffix would need the *high* bits of
the key, which cost much more to extract than the low ones the prefix filter
uses (see [README → Checking the prefix](README.md#checking-the-prefix-without-packing-the-key)).

### I don't care which digits — I just want a "pretty" key.
Use the repeat rules; they combine with each other and with prefixes:

```
./meshcore-vanity --repeat-nibble 9            # 000000000…, AAAAAAAAA…
./meshcore-vanity --repeat-byte 5              # ABABABABAB…, also 5555555555…
./meshcore-vanity --repeat-nibble 9 --repeat-byte 5 cafe
```

"Any of 16 digits" is worth one digit, "any of 256 bytes" one whole byte: nine
identical digits cost as much as a specific 8-digit prefix, and so do five
identical bytes. The check is as cheap as a single prefix, unlike listing the
16 or 256 variants as prefixes (~20% of the throughput, or most of it). The
`Repeat:` line of a hit gives the run the key actually has, which may be longer
than asked.

### What is the longest prefix?
64 hex digits, the whole key. The GPU only pre-filters on the first 16 digits;
every candidate that passes is recomputed on the host as the full compressed key
and checked against the whole criterion. No one will ever find more than ~16
anyway.

### Is my search reproducible / can it repeat work?
No and no. Every GPU thread starts from its own random base scalar and walks a
contiguous run from it, never revisiting a candidate; runs of different threads
start at random points of a 2²⁵¹-sized space, so they do not meet. Every run of
the program draws new bases.

### Are several keys from one run related?
No. A key is its thread's base plus a known small offset, so two keys from one
base would be trivially related. That is why a thread gives at most one key per
base: after a hit it gets a fresh random base. `--limit N` therefore prints N
independent keys.

### `version 'GLIBC_2.38' not found` on Ubuntu 22.04?
The released binary is built to keep its highest glibc symbol at 2.34, so it runs
on Ubuntu 22.04 (glibc 2.35). If you build it yourself on a newer distro and hit
this, it's from `atoi`/`atol` being redirected to `__isoc23_strtol@GLIBC_2.38` —
this project uses a private `parse_long` to avoid exactly that.

### `forward compatibility was attempted on non supported HW` (CUDA error 804)?
Your environment has NVIDIA's `cuda-compat` layer on the library path, which only
applies to datacenter GPUs. On a consumer GPU, point the loader at the normal
driver libs instead, e.g.
`LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./meshcore-vanity ...`, or disable
`/etc/ld.so.conf.d/000_cuda-compat.conf`.

### `out of memory` at kernel launch?
Large windows reserve a lot of GPU memory — the driver pins per-thread local
memory for the SM's full thread capacity (`SM_count × maxThreadsPerSM × W·20`
bytes), which for big `W` can be several GB. The default caps itself at `W=2048`
(~1 GB on a 16-SM GPU) and falls back further on OOM, so you should only see this
after asking for a big `--window` explicitly. If it still fails, lower `--blocks`
or set a smaller `--window` (e.g. `128` or `64`). Run `--benchmark` to see which
windows actually fit your GPU (unfitting ones show `OOM/skip`).

### Which `--window` (and `--tpb`) should I use?
Mostly you should leave `--window` alone. Since a thread now carries its walk
across launches, throughput barely depends on the window: past ~1024 the only
thing left to amortise is one field inversion per window, worth under 1.3% in
total, while the memory reserve keeps growing with `W`. The default caps at 2048
for that reason. If you do want the last percent, run `--benchmark`: it times
every window that fits (~1s each) with the memory reserve, then sweeps the block
size over the top two windows and prints the exact `--window`/`--tpb` to use.
Don't skip the `--tpb` part: 128 or 384 usually beat the 256 default. 384 is the hard ceiling (the kernel is pinned
to 168 registers per thread so a 12-warp block still fits the 64K-registers-
per-block budget); anything larger is clamped with a message.

### Is it safe? Is my private key exposed?
The private key is generated locally on your machine and only printed to stdout —
nothing is sent anywhere. Both halves of it (the scalar, via the thread's base,
and the 32-byte signing half) come straight from the operating system's
cryptographic random source (`getrandom` on Linux, `BCryptGenRandom` on
Windows). The signing half matters as much as the scalar: MeshCore derives every
signature's nonce from it, so anyone who knew it could recover the scalar from a
single signature. As with any vanity generator: keep the output private, and
prefer keys you generated yourself over any shared by a third party.
