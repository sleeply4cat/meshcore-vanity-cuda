# FAQ

### What does this do?
It brute-forces MeshCore Ed25519 keypairs on the GPU until the public key starts
with a hex prefix you choose (a "vanity" key). It's a from-scratch CUDA rewrite
of the OpenCL `nano-vanity`/MeshCore fork, ~49× faster on the same GPU.

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
average. As a rough guide at ~300 Mkeys/s: 6 nibbles ≈ instant, 8 nibbles ≈
~15 s, 10 nibbles ≈ ~1 h, 12 nibbles ≈ ~10 days. The tool prints
`Estimated attempts: 2^bits` at startup.

### Can I match multiple prefixes, a suffix, or a regex?
Not currently — one prefix per run. The per-candidate cost is dominated by the
field arithmetic, so extra matching is cheap in principle, but it is not
implemented. Run separate instances for separate prefixes (one GPU handles one
instance at full speed; two instances on one GPU roughly halve each).

### Why is the max prefix 63 nibbles and not 64?
The in-kernel filter matches the low 255 bits of the public key (the `y`
coordinate); bit 255 is the `x`-parity sign and isn't used for filtering. The
displayed key is always the full, correct 256-bit compressed key. 63 nibbles is
irrelevant in practice — no one searches prefixes that long.

### Is my search reproducible / can it repeat work?
The base counter starts from a random clamped value and advances by exactly the
span covered each launch, so intervals are contiguous and never overlap within a
process — no batch is ever recomputed. Different runs start from a fresh random
base.

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
memory for the SM's full thread capacity (`SM_count × maxThreadsPerSM × W·40`
bytes), which for big `W` can be several GB. The default auto-fits the window to
free VRAM and falls back to smaller windows on OOM. If it still fails, lower
`--blocks` or set a smaller `--window` (e.g. `128` or `64`). Run `--benchmark`
to see which windows actually fit your GPU (unfitting ones show `OOM/skip`).

### Which `--window` (and `--tpb`) should I use?
Run `--benchmark`: it times every window that fits (~1s each) with the memory
reserve, then sweeps the block size at the fastest window and prints the exact
`--window`/`--tpb` to use. Gains above ~1024 are hardware-dependent, so measure
rather than assume bigger is better.

### Is it safe? Is my private key exposed?
The private key is generated locally on your machine and only printed to stdout —
nothing is sent anywhere. As with any vanity generator: keep the output private,
and prefer keys you generated yourself over any shared by a third party.
