// meshcore-vanity-cuda — GPU vanity public-key search for MeshCore.
//
// Algorithm (see README): pubkey(s+8) = pubkey(s) + 8*B, so instead of a full
// fixed-base scalar multiplication per candidate we do ONE point addition
// (P += 8B) and amortize the field inversion over a window of WINDOW
// candidates with Montgomery's batch-inversion trick.
//
// Each thread g owns candidates s = base + (g*WINDOW + j)*8, j in [0,WINDOW).
// One fixed-base multiply computes the *centre* of that span, then the window is
// walked outwards in +/-i pairs: since -Q = (-x, y), one table entry i*D yields
// both s = centre +/- 8i, so a window of W candidates needs only W/2 table
// entries and W/2 batch-inversion slots.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <random>
#include <chrono>

#include "ed25519.cuh"

// Window = candidates per thread per launch (batch size). It is chosen at
// runtime (--window, or auto-fit to VRAM); the kernel is templated on it so the
// per-thread buffers stay compile-time sized.
//
// Supported windows, descending. The single per-thread buffer (pref) holds one
// prefix product per +/-i PAIR, i.e. W/2 elements = W*20 bytes of local memory,
// so W=16384 = 320 KB/thread, still under the 512 KB per-thread local limit
// (numerators and denominators are recomputed, not stored). Bigger windows
// amortize the two per-window inversions over more candidates but reserve much
// more VRAM (16384 needs a 16 GB+ GPU); pick them with --window. The 1.5x
// "half" windows (1536/3072/6144/12288) fill the gaps between powers of two so
// a size can be chosen closer to what VRAM allows; all are multiples of 512
// (a warp is 32, a default block 256) for clean local-frame/table alignment.
static const int kWindows[] = {16384, 12288, 8192, 6144, 4096, 3072, 2048,
                               1536, 1024, 512, 256, 128, 64};

#define RESULT_CAP 4096      // max matches recorded per launch

// -------------------------------------------------------------------------
// 256-bit little-endian helpers (host + device)
// -------------------------------------------------------------------------

// r = a + add, treating a as a 32-byte little-endian integer.
__host__ __device__ static void scalar_add_u64(uint8_t r[32], const uint8_t a[32], uint64_t add) {
    uint64_t carry = add;
    for (int i = 0; i < 32; i++) {
        uint64_t s = (uint64_t)a[i] + (carry & 0xff);
        r[i] = (uint8_t)s;
        carry = (carry >> 8) + (s >> 8);
    }
}

// MeshCore/Ed25519 clamp (idempotent on the values we walk).
__host__ __device__ static void clamp_scalar(uint8_t s[32]) {
    s[0]  &= 248;
    s[31] &= 127;
    s[31] |= 64;
}

// -------------------------------------------------------------------------
// Device: read D = 8*B in niels form. ge25519_niels_base_multiples[k] holds
// the niels encoding of (k+1)*B, so entry [7] == 8*B — no computation needed.
// -------------------------------------------------------------------------
__device__ static void load_step_8B(ge25519_niels *D) {
    curve25519_expand(D->ysubx, ge25519_niels_base_multiples[7] + 0);
    curve25519_expand(D->xaddy, ge25519_niels_base_multiples[7] + 32);
    curve25519_expand(D->t2d,   ge25519_niels_base_multiples[7] + 64);
    // The base-multiples table stores t2d = 2*x*y; ge25519_nielsadd2 applied to
    // a standard extended point expects 2*d*x*y (this is why donna multiplies
    // the b[0] niels by ecd). Our accumulator is a clean point from
    // scalarmult, so pre-multiply once here.
    curve25519_mul_const(D->t2d, D->t2d, ge25519_ecd);
}

// P = s*B  (s given as 32 clamped LE bytes), returned in extended coords.
__device__ static void scalar_to_point(ge25519 *P, const uint8_t s[32]) {
    bignum256modm sm;
    expand256_modm(sm, s, 32);
    ge25519_scalarmult_base_niels(P, sm);
}

// Extended -> affine (x,y) = (X/Z, Y/Z). One inversion.
__device__ static void ge_to_affine(bignum25519 ax, bignum25519 ay, const ge25519 *P) {
    bignum25519 zi;
    curve25519_recip(zi, P->z);
    curve25519_mul(ax, P->x, zi);
    curve25519_mul(ay, P->y, zi);
}

// -------------------------------------------------------------------------
// Precompute the step table: affine coords of i*D (D = 8B) for i in [1,WINDOW/2],
// plus P_i = x_i*y_i (so the hot loop needs no extra mul for the denominator).
// Only half a window is tabulated because entry i serves both s = centre + 8i
// and s = centre - 8i (negating a point flips only x).
// These points are identical for every thread, so this runs once and the main
// kernel just reads the table (broadcast across the warp). Single thread; the
// per-point inversion cost is one-time and negligible.
// -------------------------------------------------------------------------
__global__ void build_step_table_kernel(bignum25519 *gx, bignum25519 *gy, bignum25519 *gp, int window) {
    ge25519_niels D; load_step_8B(&D);
    uint8_t eight[32]; for (int k = 0; k < 32; k++) eight[k] = 0; eight[0] = 8;
    ge25519 Q; scalar_to_point(&Q, eight);   // Q = 8B = 1*D
    const int n = window / 2;
    for (int i = 1; i <= n; i++) {
        ge_to_affine(gx[i], gy[i], &Q);
        curve25519_mul(gp[i], gx[i], gy[i]);
        if (i < n) ge25519_nielsadd2(&Q, &D);            // Q += D
    }
}

// -------------------------------------------------------------------------
// Low 8 bytes of the compressed encoding of a field element, as one uint64.
//
// The filter only ever looks at the first bytes of the key and the encoding is
// little-endian, so packing all 32 bytes per candidate is waste. The
// *reduction* cannot be shortened — radix 2^25.5 folds the top of the value
// back into limb 0 via 2^255 == 19, so the low bits genuinely depend on every
// limb — but the packing can: after the same canonicalisation
// curve25519_contract performs, limbs 0..2 cover bits 0..76, and bits 0..63 are
// exactly f0 | f1<<26 | f2<<51.
//
// This mirrors curve25519_contract's reduction; --selftest cross-checks the two
// against each other on random and edge-case inputs so they cannot drift apart.
// -------------------------------------------------------------------------
__device__ static uint64_t curve25519_contract_lo64(const bignum25519 in) {
    bignum25519 f;
    curve25519_copy(f, in);

#define LO64_CARRY() \
    f[1] += f[0] >> 26; f[0] &= reduce_mask_26; \
    f[2] += f[1] >> 25; f[1] &= reduce_mask_25; \
    f[3] += f[2] >> 26; f[2] &= reduce_mask_26; \
    f[4] += f[3] >> 25; f[3] &= reduce_mask_25; \
    f[5] += f[4] >> 26; f[4] &= reduce_mask_26; \
    f[6] += f[5] >> 25; f[5] &= reduce_mask_25; \
    f[7] += f[6] >> 26; f[6] &= reduce_mask_26; \
    f[8] += f[7] >> 25; f[7] &= reduce_mask_25; \
    f[9] += f[8] >> 26; f[8] &= reduce_mask_26;
#define LO64_CARRY_FULL() LO64_CARRY() f[0] += 19 * (f[9] >> 25); f[9] &= reduce_mask_25;

    LO64_CARRY_FULL()
    LO64_CARRY_FULL()
    // Now 0 <= f < 2^255. Offset by 19 to separate the two canonical cases,
    // add 2^255, carry, and drop the borrow — exactly as curve25519_contract.
    f[0] += 19;
    LO64_CARRY_FULL()
    f[0] += (reduce_mask_26 + 1) - 19;
    f[1] += (reduce_mask_25 + 1) - 1;
    f[2] += (reduce_mask_26 + 1) - 1;
    f[3] += (reduce_mask_25 + 1) - 1;
    f[4] += (reduce_mask_26 + 1) - 1;
    f[5] += (reduce_mask_25 + 1) - 1;
    f[6] += (reduce_mask_26 + 1) - 1;
    f[7] += (reduce_mask_25 + 1) - 1;
    f[8] += (reduce_mask_26 + 1) - 1;
    f[9] += (reduce_mask_25 + 1) - 1;
    LO64_CARRY()
    // (f[9] is masked off in contract here; bits >= 255 do not reach the low 64.)
#undef LO64_CARRY_FULL
#undef LO64_CARRY

    return (uint64_t)f[0] | ((uint64_t)f[1] << 26) | ((uint64_t)f[2] << 51);
}

// -------------------------------------------------------------------------
// Main search kernel — affine batched-addition walk in +/-i pairs (y only).
//
// For candidate i (scalar s0 + i*8) the point is P0 + i*D. Using the complete
// twisted-Edwards (a=-1) addition and keeping only y:
//     y_i = (x0*x_i + y0*y_i) / (1 - d*x0*y0 * x_i*y_i)
// where (x0,y0) is the window-CENTRE point (one fixed-base multiply per thread)
// and (x_i, y_i, P_i=x_i*y_i) come from the shared precomputed step table.
//
// Negating a point flips only x (-Q = (-x, y)), so ONE table entry gives two
// candidates. With A = x0*x_i, B = y0*y_i and C = K*P_i (K = d*x0*y0, computed
// once per thread) the pair costs three muls up front:
//     centre + 8i:  y = (B + A) / (1 - C)
//     centre - 8i:  y = (B - A) / (1 + C)
// and, crucially, the two denominators share one batch-inversion slot because
//     (1 - C)*(1 + C) = 1 - C^2      (one squaring)
// which is then split back with two muls: 1/(1-C) = (1+C)*inv, and vice versa.
// That is ~10 muls + 2 squarings per PAIR (~5.5 per candidate, vs ~8 for the
// one-sided walk) and halves both the local-memory buffer and the step table.
//
// Register/occupancy tuning (maxrregcount, __launch_bounds__) was benchmarked
// and is neutral-to-worse: ALU-bound, high ILP saturates the int units even at
// low occupancy, so plain launch is best. (Re-measured for the pair walk, whose
// wider live set pushes the kernel to ~160 registers: forcing it back to 128 to
// unlock tpb=512 costs more than the extra resident warps return.) The 160-reg
// footprint does cap the block size below 512 on a 64K-register-per-block GPU —
// --benchmark sweeps tpb, so it lands on a launchable peak by itself.
template <int W>
__global__ void vanity_kernel(const uint8_t *__restrict__ base,
                              const bignum25519 *__restrict__ gx,
                              const bignum25519 *__restrict__ gy,
                              const bignum25519 *__restrict__ gp,
                              const uint8_t *__restrict__ req,
                              const uint8_t *__restrict__ mask,
                              int prefix_len,
                              unsigned long long *__restrict__ out_count,
                              unsigned long long *__restrict__ out_units) {
    const int H = W / 2;
    const unsigned long long gid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    // The thread still owns units [gid*W, gid*W + W); it just works outwards
    // from the middle of that span, so the covered range (and hence the clamp
    // and no-overlap analysis) is exactly as before.
    const unsigned long long centre_unit = gid * (unsigned long long)W + (unsigned long long)H;

    // Pack the first (up to) 8 prefix bytes into one masked 64-bit comparison.
    // Keeping them as byte arrays would cost a dynamically indexed register
    // array in the innermost loop, which nvcc lowers to a select chain — worth
    // ~5% of total throughput. Longer prefixes fall back to a byte compare that
    // only a 1-in-2^64 candidate ever reaches.
    uint64_t req8 = 0, mask8 = 0;
    for (int i = 0; i < 8 && i < prefix_len; i++) {
        req8  |= (uint64_t)req[i]  << (8 * i);
        mask8 |= (uint64_t)mask[i] << (8 * i);
    }

    // Window centre scalar = base + centre_unit*8, then its affine point.
    uint8_t s0[32];
    scalar_add_u64(s0, base, centre_unit * 8ULL);
    clamp_scalar(s0);

    ge25519 P;
    scalar_to_point(&P, s0);
    bignum25519 x0, y0;
    ge_to_affine(x0, y0, &P);

    // K = d * x0 * y0  (per-thread constant for the denominator).
    bignum25519 K;
    curve25519_mul_const(K, x0, ge25519_ecd);
    curve25519_mul(K, K, y0);

    bignum25519 one; for (int k = 0; k < 10; k++) one[k] = 0; one[0] = 1;

    auto check_and_record = [&](const bignum25519 y, unsigned long long unit) {
        // Matches the low 255 bits of y (bit 255 = x parity is ignored; the
        // host recomputes the full compressed key for any hit).
        if ((curve25519_contract_lo64(y) ^ req8) & mask8) return;
        if (prefix_len > 8) {                          // effectively never taken
            unsigned char pub[32];
            curve25519_contract(pub, y);
            for (int i = 8; i < prefix_len; i++)
                if ((pub[i] & mask[i]) != req[i]) return;
        }
        unsigned long long slot = atomicAdd(out_count, 1ULL);
        if (slot < RESULT_CAP) out_units[slot] = unit;
    };

    // The centre candidate needs no table entry and no inversion: y = y0.
    check_and_record(y0, centre_unit);

    // Pairs i=1..H (H = W/2): one batch-inversion slot per pair, holding the
    // PRODUCT of the two denominators. ONLY the prefix products are stored;
    // C = K*gp_i, the two denominators and the two numerators are all
    // recomputed in the backward pass from x0,y0,K and the shared table, so a
    // single H-element buffer holds the whole window. Recomputing costs one
    // extra mul + one extra squaring per pair, a cheap trade for removing the
    // local-memory (DRAM) store+load streams and for fitting larger windows.
    bignum25519 pref[W / 2];
    bignum25519 acc; curve25519_copy(acc, one);
    for (int i = 1; i <= H; i++) {
        bignum25519 c, cc, prod;
        curve25519_mul(c, K, gp[i]);                   // C = d x0 y0 x_i y_i
        curve25519_square(cc, c);
        curve25519_sub_reduce(prod, one, cc);          // (1-C)(1+C) = 1 - C^2
        curve25519_copy(pref[i - 1], acc);             // prefix product
        curve25519_mul(acc, acc, prod);
    }
    curve25519_recip(acc, acc);                        // 1 / prod(1 - C^2)

    for (int i = H; i >= 1; i--) {
        bignum25519 c, cc, prod, invprod, a, b, num, den, y;
        curve25519_mul(invprod, acc, pref[i - 1]);     // 1 / ((1-C)(1+C))
        curve25519_mul(c, K, gp[i]);                   // recompute C
        curve25519_square(cc, c);
        curve25519_sub_reduce(prod, one, cc);
        curve25519_mul(acc, acc, prod);                // strip this pair
        curve25519_mul(a, x0, gx[i]);                  // A = x0 x_i
        curve25519_mul(b, y0, gy[i]);                  // B = y0 y_i

        // centre - 8i : y = (B - A) / (1 + C),  1/(1+C) = (1-C) * invprod
        curve25519_sub_reduce(den, one, c);
        curve25519_mul(den, den, invprod);
        curve25519_sub_reduce(num, b, a);
        curve25519_mul(y, num, den);
        check_and_record(y, centre_unit - (unsigned long long)i);

        // centre + 8i : y = (B + A) / (1 - C),  1/(1-C) = (1+C) * invprod.
        // i == H would land on the next thread's first unit, so skip it — the
        // span stays exactly W units wide and threads never overlap.
        if (i < H) {
            curve25519_add_reduce(den, one, c);
            curve25519_mul(den, den, invprod);
            curve25519_add_reduce(num, b, a);
            curve25519_mul(y, num, den);
            check_and_record(y, centre_unit + (unsigned long long)i);
        }
    }
}

// -------------------------------------------------------------------------
// Host-callable single-key pack: full compressed pubkey (incl. parity) for a
// given clamped scalar. Used to verify/display hits.
// -------------------------------------------------------------------------
// Selftest helper: contract both ways so the host can check that the fast
// low-64-bit packing agrees with donna's full 32-byte contract.
__global__ void contract_lo64_kernel(const bignum25519 *__restrict__ in, int n,
                                     uint8_t *__restrict__ out32,
                                     unsigned long long *__restrict__ out_lo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curve25519_contract(out32 + i * 32, in[i]);
    out_lo[i] = curve25519_contract_lo64(in[i]);
}

__global__ void pack_one_kernel(const uint8_t *__restrict__ scalar, uint8_t *__restrict__ pub) {
    ge25519 P;
    scalar_to_point(&P, scalar);
    ge25519_pack(pub, &P);
}

// -------------------------------------------------------------------------
// Self-test kernel: verifies the incremental identity
//   scalarmult(s+8) == scalarmult(s) + 8B
// for a batch of scalars, and packs scalarmult(s) so the host can check a KAT.
// results[i]=1 on identity match. pub0 = packed scalarmult(scalars[0]).
// -------------------------------------------------------------------------
__global__ void selftest_kernel(const uint8_t *__restrict__ scalars, int n,
                                uint8_t *__restrict__ ok, uint8_t *__restrict__ pub0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t *s = scalars + (size_t)i * 32;

    uint8_t s8[32];
    scalar_add_u64(s8, s, 8ULL);

    ge25519 P, Pfull8;
    scalar_to_point(&P, s);           // P = s*B
    scalar_to_point(&Pfull8, s8);     // (s+8)*B via full multiply

    if (i == 0) ge25519_pack(pub0, &P);

    ge25519_niels D; load_step_8B(&D);
    ge25519_nielsadd2(&P, &D);        // P = s*B + 8B  (incremental)

    uint8_t a[32], b[32];
    ge25519_pack(a, &P);
    ge25519_pack(b, &Pfull8);
    bool same = true;
    for (int k = 0; k < 32; k++) if (a[k] != b[k]) { same = false; break; }
    ok[i] = same ? 1 : 0;
}

// =========================================================================
// Host code
// =========================================================================

static void cuda_check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

// Parse a hex prefix into (req, mask) byte vectors, like the reference
// build_matcher_from_hex. Returns prefix_len in bytes.
static int build_matcher(const std::string &hex, std::vector<uint8_t> &req,
                         std::vector<uint8_t> &mask) {
    std::vector<uint8_t> nibbles;
    for (char c : hex) {
        char lc = (char)tolower((unsigned char)c);
        int v;
        if (lc >= '0' && lc <= '9') v = lc - '0';
        else if (lc >= 'a' && lc <= 'f') v = 10 + lc - 'a';
        else { fprintf(stderr, "Invalid hex character in prefix: '%c'\n", c); exit(1); }
        nibbles.push_back((uint8_t)v);
    }
    int full_bytes = (int)(nibbles.size() + 1) / 2;
    req.assign(full_bytes, 0);
    mask.assign(full_bytes, 0);
    for (size_t i = 0; i < nibbles.size(); i++) {
        int bi = (int)i / 2;
        if (i % 2 == 0) { req[bi] |= nibbles[i] << 4; mask[bi] |= 0xF0; }
        else            { req[bi] |= nibbles[i];      mask[bi] |= 0x0F; }
    }
    return full_bytes;
}

static std::string hex_upper(const uint8_t *p, int n) {
    static const char *H = "0123456789ABCDEF";
    std::string s;
    s.reserve(n * 2);
    for (int i = 0; i < n; i++) { s += H[p[i] >> 4]; s += H[p[i] & 0xf]; }
    return s;
}

// Minimal base-10 parser for CLI integers. Deliberately avoids atoi/atol/strtol:
// with a recent GCC/glibc those get redirected to __isoc23_strtol (GLIBC_2.38),
// which would break the binary on older distros (e.g. Ubuntu 22.04 / glibc 2.35).
// This keeps the highest required glibc symbol at 2.34.
static long parse_long(const char *s) {
    while (*s == ' ' || *s == '\t') s++;
    long sign = 1;
    if (*s == '+') s++;
    else if (*s == '-') { sign = -1; s++; }
    long v = 0;
    for (; *s >= '0' && *s <= '9'; s++) v = v * 10 + (*s - '0');
    return sign * v;
}

static void fill_random(uint8_t *p, int n) {
    static std::random_device rd;
    static std::mt19937_64 gen(((uint64_t)rd() << 32) ^ rd());
    for (int i = 0; i < n; i++) p[i] = (uint8_t)(gen() & 0xff);
}

// Pack one clamped scalar -> full compressed pubkey (device round-trip).
static void pack_one(const uint8_t scalar[32], uint8_t pub[32],
                    uint8_t *d_scalar, uint8_t *d_pub) {
    cuda_check(cudaMemcpy(d_scalar, scalar, 32, cudaMemcpyHostToDevice), "memcpy scalar");
    pack_one_kernel<<<1, 1>>>(d_scalar, d_pub);
    cuda_check(cudaGetLastError(), "pack_one launch");
    cuda_check(cudaMemcpy(pub, d_pub, 32, cudaMemcpyDeviceToHost), "memcpy pub");
}

// Kernel launch args, bundled so the window dispatch stays readable.
struct LaunchArgs {
    const uint8_t *base; const bignum25519 *gx, *gy, *gp;
    const uint8_t *req, *mask; int prefix_len;
    unsigned long long *count, *units;
};

// Launch the vanity kernel instantiation for a runtime window. Returns the
// launch error (e.g. cudaErrorMemoryAllocation if the local frame won't fit).
static cudaError_t launch_vanity(int window, int blocks, int tpb, const LaunchArgs &a) {
#define LV(W) vanity_kernel<W><<<blocks, tpb>>>(a.base, a.gx, a.gy, a.gp, a.req, a.mask, \
                                                a.prefix_len, a.count, a.units)
    switch (window) {
        case 16384: LV(16384); break;
        case 12288: LV(12288); break;
        case 8192: LV(8192); break;
        case 6144: LV(6144); break;
        case 4096: LV(4096); break;
        case 3072: LV(3072); break;
        case 2048: LV(2048); break;
        case 1536: LV(1536); break;
        case 1024: LV(1024); break;
        case 512:  LV(512);  break;
        case 256:  LV(256);  break;
        case 128:  LV(128);  break;
        case 64:   LV(64);   break;
        default: return cudaErrorInvalidValue;
    }
#undef LV
    return cudaGetLastError();
}

// All vanity_kernel<W> share one signature, so a plain const void* pointer lets
// the runtime introspection helpers (attributes, occupancy) take a single path.
static const void *vanity_kernel_ptr(int window) {
    switch (window) {
        case 16384: return (const void *)vanity_kernel<16384>;
        case 12288: return (const void *)vanity_kernel<12288>;
        case 8192: return (const void *)vanity_kernel<8192>;
        case 6144: return (const void *)vanity_kernel<6144>;
        case 4096: return (const void *)vanity_kernel<4096>;
        case 3072: return (const void *)vanity_kernel<3072>;
        case 2048: return (const void *)vanity_kernel<2048>;
        case 1536: return (const void *)vanity_kernel<1536>;
        case 1024: return (const void *)vanity_kernel<1024>;
        case 512:  return (const void *)vanity_kernel<512>;
        case 256:  return (const void *)vanity_kernel<256>;
        case 128:  return (const void *)vanity_kernel<128>;
        default:   return (const void *)vanity_kernel<64>;
    }
}

// Per-thread local memory (bytes) the given window instantiation needs.
static size_t window_local_bytes(int window) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, vanity_kernel_ptr(window)) != cudaSuccess)
        return (size_t)window * 20;   // W/2 prefix products x 40 bytes
    return fa.localSizeBytes;
}

// Worst-case per-thread local-memory reserve for a window. This is inherent,
// documented CUDA behavior (not a bug): the driver provisions a local-memory
// backing store with a private slot for every thread that could ever be
// resident, sized by the SM's *absolute* thread capacity (maxThreadsPerSM),
// independent of the kernel's actual occupancy. Verified two ways: capping a
// probe kernel's occupancy 100%->33% left the reservation unchanged, and this
// kernel (register-limited to ~33%) still OOMs window 2048 at the full
// numSM*maxThreadsPerSM*240KB. The only lever is loc/thr itself, not occupancy.
static size_t window_reserve_bytes(int window, int numSM, int maxThreadsSM) {
    return (size_t)numSM * maxThreadsSM * window_local_bytes(window);
}

// Largest supported window whose local-memory reserve fits free VRAM with a
// margin.
static int auto_window() {
    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);
    int numSM = 1, maxThreadsSM = 1024, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&maxThreadsSM, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    for (int w : kWindows)
        if ((double)window_reserve_bytes(w, numSM, maxThreadsSM) < 0.85 * (double)freeB)
            return w;
    return kWindows[sizeof(kWindows) / sizeof(kWindows[0]) - 1];  // smallest
}

static int nearest_window(int req) {
    int best = kWindows[0];
    for (int w : kWindows) if (w <= req) { best = w; break; }  // largest supported <= req
    // if req smaller than all, fall through to smallest
    if (req < kWindows[sizeof(kWindows)/sizeof(kWindows[0]) - 1])
        best = kWindows[sizeof(kWindows)/sizeof(kWindows[0]) - 1];
    return best;
}

static int run_selftest();
static int run_benchmark(int blocks, int tpb);

int main(int argc, char **argv) {
    std::string prefix;
    long limit = 1;
    int blocks = 512;
    int tpb = 256;
    int device = 0;
    int window = 0;              // 0 = auto-fit to VRAM
    bool progress = true;
    bool selftest = false;
    bool benchmark = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char *n) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", n); exit(1); }
            return argv[++i];
        };
        if (a == "--selftest") selftest = true;
        else if (a == "--benchmark" || a == "--bench") benchmark = true;
        else if (a == "-l" || a == "--limit") limit = parse_long(need("--limit"));
        else if (a == "--blocks") blocks = (int)parse_long(need("--blocks"));
        else if (a == "--tpb") tpb = (int)parse_long(need("--tpb"));
        else if (a == "-d" || a == "--device") device = (int)parse_long(need("--device"));
        else if (a == "-w" || a == "--window") window = (int)parse_long(need("--window"));
        else if (a == "--no-progress") progress = false;
        else if (a == "-h" || a == "--help") {
            printf("Usage: %s <HEX_PREFIX> [options]\n"
                   "  -l, --limit N     stop after N matches (0 = infinite) [1]\n"
                   "  -w, --window N    batch/thr: 64..16384 incl. 1536/3072/6144/12288 [auto VRAM]\n"
                   "                    bigger = faster but more GPU memory\n"
                   "      --blocks N    CUDA blocks [512]\n"
                   "      --tpb N       threads per block [256]\n"
                   "  -d, --device I    CUDA device index [0]\n"
                   "      --no-progress suppress progress output\n"
                   "      --selftest    run correctness self-tests and exit\n"
                   "      --benchmark   sweep window + block size, print the fastest, and exit\n",
                   argv[0]);
            return 0;
        }
        else if (!a.empty() && a[0] == '-') { fprintf(stderr, "Unknown option %s\n", a.c_str()); return 1; }
        else prefix = a;
    }

    cuda_check(cudaSetDevice(device), "setDevice");
    // Block (sleep) the host thread while waiting on the GPU instead of the
    // default busy-wait spin, which otherwise pegs one CPU core at 100% and
    // eats into the shared laptop power/thermal budget (lowering GPU boost).
    cuda_check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync), "setDeviceFlags");

    if (selftest) return run_selftest();
    if (benchmark) return run_benchmark(blocks, tpb);

    if (prefix.empty() || prefix.size() > 64) {
        fprintf(stderr, "Prefix must be 1-64 hex characters. Use --help.\n");
        return 1;
    }

    // Resolve the window: explicit --window (snapped to a supported size) or
    // auto-fit to free VRAM.
    if (window == 0) {
        window = auto_window();
        fprintf(stderr, "Window: auto-selected %d (largest that fits; run --benchmark "
                        "to find the fastest for your GPU)\n", window);
    } else {
        int snap = nearest_window(window);
        if (snap != window) {
            // Print the supported list from kWindows itself so it never drifts.
            std::string sup;
            for (int i = (int)(sizeof(kWindows) / sizeof(kWindows[0])) - 1; i >= 0; i--)
                sup += (sup.empty() ? "" : "/") + std::to_string(kWindows[i]);
            fprintf(stderr, "Window: %d not supported, using %d (supported: %s)\n",
                    window, snap, sup.c_str());
        }
        window = snap;
    }

    std::vector<uint8_t> req, mask;
    int prefix_len = build_matcher(prefix, req, mask);

    const unsigned long long threads = (unsigned long long)blocks * tpb;
    unsigned long long per_launch = threads * (unsigned long long)window;

    fprintf(stderr, "Searching for pubkey prefix: ");
    for (char c : prefix) fputc(toupper((unsigned char)c), stderr);
    fprintf(stderr, "\n  Req:  %s\n  Mask: %s\n",
            hex_upper(req.data(), prefix_len).c_str(),
            hex_upper(mask.data(), prefix_len).c_str());
    // Expected attempts = 2^(bits set in mask).
    int bits = 0; for (uint8_t m : mask) bits += __builtin_popcount(m);
    fprintf(stderr, "Estimated attempts: 2^%d\n", bits);
    fprintf(stderr, "Grid: %d blocks x %d threads, window %d => %llu candidates/launch\n",
            blocks, tpb, window, per_launch);

    // Device buffers.
    uint8_t *d_base, *d_req, *d_mask, *d_scalar, *d_pub;
    unsigned long long *d_count, *d_units;
    cuda_check(cudaMalloc(&d_base, 32), "malloc base");
    cuda_check(cudaMalloc(&d_req, prefix_len ? prefix_len : 1), "malloc req");
    cuda_check(cudaMalloc(&d_mask, prefix_len ? prefix_len : 1), "malloc mask");
    cuda_check(cudaMalloc(&d_count, sizeof(unsigned long long)), "malloc count");
    cuda_check(cudaMalloc(&d_units, sizeof(unsigned long long) * RESULT_CAP), "malloc units");
    cuda_check(cudaMalloc(&d_scalar, 32), "malloc scalar");
    cuda_check(cudaMalloc(&d_pub, 32), "malloc pub");

    // Shared precomputed step table i*D (affine x, y and x*y), i in [1,window/2].
    const size_t tbl = (size_t)(window / 2) + 1;
    bignum25519 *d_gx, *d_gy, *d_gp;
    cuda_check(cudaMalloc(&d_gx, sizeof(bignum25519) * tbl), "malloc gx");
    cuda_check(cudaMalloc(&d_gy, sizeof(bignum25519) * tbl), "malloc gy");
    cuda_check(cudaMalloc(&d_gp, sizeof(bignum25519) * tbl), "malloc gp");
    build_step_table_kernel<<<1, 1>>>(d_gx, d_gy, d_gp, window);
    cuda_check(cudaGetLastError(), "build_step_table launch");
    cuda_check(cudaDeviceSynchronize(), "build_step_table sync");
    cuda_check(cudaMemcpy(d_req, req.data(), prefix_len, cudaMemcpyHostToDevice), "memcpy req");
    cuda_check(cudaMemcpy(d_mask, mask.data(), prefix_len, cudaMemcpyHostToDevice), "memcpy mask");

    // Random, once-only base counter (advanced deterministically per launch).
    uint8_t base[32];
    fill_random(base, 32);
    clamp_scalar(base);

    long found = 0;
    unsigned long long attempts = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto tlast = t0;

    while (true) {
        cuda_check(cudaMemcpy(d_base, base, 32, cudaMemcpyHostToDevice), "memcpy base");
        cuda_check(cudaMemset(d_count, 0, sizeof(unsigned long long)), "memset count");

        LaunchArgs la{d_base, d_gx, d_gy, d_gp, d_req, d_mask, prefix_len, d_count, d_units};
        cudaError_t le = launch_vanity(window, blocks, tpb, la);
        if (le == cudaErrorMemoryAllocation) {
            // Auto-fall back to a smaller window (shrinks the per-thread local
            // frame) until it fits, so one binary works on any GPU.
            int smaller = 0;
            for (int w : kWindows) if (w < window) { smaller = w; break; }
            if (smaller) {
                fprintf(stderr, "\nWindow %d out of memory; falling back to %d.\n", window, smaller);
                window = smaller;
                per_launch = threads * (unsigned long long)window;
                cudaGetLastError();  // clear
                continue;
            }
            fprintf(stderr, "\nCUDA out of memory even at the smallest window (%d). "
                            "Try fewer --blocks.\n", window);
            exit(1);
        }
        cuda_check(le, "kernel launch");
        cuda_check(cudaDeviceSynchronize(), "sync");

        unsigned long long count = 0;
        cuda_check(cudaMemcpy(&count, d_count, sizeof(count), cudaMemcpyDeviceToHost), "copy count");
        unsigned long long n_units = count < RESULT_CAP ? count : RESULT_CAP;
        std::vector<unsigned long long> units(n_units);
        if (n_units)
            cuda_check(cudaMemcpy(units.data(), d_units,
                                  sizeof(unsigned long long) * n_units, cudaMemcpyDeviceToHost),
                       "copy units");

        for (unsigned long long k = 0; k < n_units; k++) {
            uint8_t scalar[32];
            scalar_add_u64(scalar, base, units[k] * 8ULL);
            clamp_scalar(scalar);

            uint8_t pub[32];
            pack_one(scalar, pub, d_scalar, d_pub);

            // Host re-check of the full prefix (defensive; also validates GPU).
            bool ok = true;
            for (int i = 0; i < prefix_len; i++)
                if ((pub[i] & mask[i]) != req[i]) { ok = false; break; }
            if (!ok) continue;

            uint8_t signing[32];
            fill_random(signing, 32);
            if (progress) fputc('\r', stderr);
            printf("\nFound matching key!\n");
            printf("Public Key:  %s\n", hex_upper(pub, 32).c_str());
            printf("Private Key: %s%s\n", hex_upper(scalar, 32).c_str(),
                   hex_upper(signing, 32).c_str());
            fflush(stdout);

            if (++found >= limit && limit != 0) goto done;
        }

        attempts += per_launch;
        // Advance base by exactly the covered span => disjoint, no repeats.
        scalar_add_u64(base, base, per_launch * 8ULL);

        if (progress) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - tlast).count() >= 0.3) {
                double secs = std::chrono::duration<double>(now - t0).count();
                double mps = secs > 0 ? attempts / secs / 1e6 : 0;
                fprintf(stderr, "\rTried %llu keys (%.1f Mkeys/s)", attempts, mps);
                tlast = now;
            }
        }
    }
done:
    fprintf(stderr, "\n");
    return 0;
}

// -------------------------------------------------------------------------
// Self-test: the fast low-64-bit packing must agree with donna's full contract
// on random field elements plus the canonicalisation edge cases (0, 1, p-1, p,
// p+1, 2^255-1), which is where a hand-rolled reduction would go wrong.
// -------------------------------------------------------------------------
static int check_contract_lo64() {
    const uint32_t m26 = (1u << 26) - 1, m25 = (1u << 25) - 1;
    std::vector<std::array<uint32_t, 10>> v;
    auto push = [&](std::array<uint32_t, 10> f) { v.push_back(f); };
    auto full = [&](uint32_t lo0) {   // limb0 = lo0, all higher limbs saturated
        std::array<uint32_t, 10> f{};
        f[0] = lo0;
        for (int k = 1; k < 10; k++) f[k] = (k & 1) ? m25 : m26;
        return f;
    };
    push({0,0,0,0,0,0,0,0,0,0});
    push({1,0,0,0,0,0,0,0,0,0});
    push(full(m26 - 20));             // p - 1
    push(full(m26 - 19));             // p     -> must canonicalise to 0
    push(full(m26 - 18));             // p + 1 -> 1
    push(full(m26));                  // 2^255 - 1
    std::mt19937 rng(12345);
    for (int i = 0; i < 4096; i++) {
        std::array<uint32_t, 10> f{};
        for (int k = 0; k < 10; k++) f[k] = rng() & ((k & 1) ? m25 : m26);
        push(f);
    }

    const int n = (int)v.size();
    uint32_t *d_in; uint8_t *d_out32; unsigned long long *d_lo;
    cuda_check(cudaMalloc(&d_in, sizeof(uint32_t) * 10 * n), "malloc lo64 in");
    cuda_check(cudaMalloc(&d_out32, 32 * n), "malloc lo64 out32");
    cuda_check(cudaMalloc(&d_lo, sizeof(unsigned long long) * n), "malloc lo64 lo");
    cuda_check(cudaMemcpy(d_in, v.data(), sizeof(uint32_t) * 10 * n, cudaMemcpyHostToDevice), "memcpy lo64");
    contract_lo64_kernel<<<(n + 127) / 128, 128>>>((const bignum25519 *)d_in, n, d_out32, d_lo);
    cuda_check(cudaGetLastError(), "lo64 launch");
    cuda_check(cudaDeviceSynchronize(), "lo64 sync");

    std::vector<uint8_t> out32(32 * n);
    std::vector<unsigned long long> lo(n);
    cuda_check(cudaMemcpy(out32.data(), d_out32, 32 * n, cudaMemcpyDeviceToHost), "copy out32");
    cuda_check(cudaMemcpy(lo.data(), d_lo, sizeof(unsigned long long) * n, cudaMemcpyDeviceToHost), "copy lo");
    cudaFree(d_in); cudaFree(d_out32); cudaFree(d_lo);

    int fails = 0;
    for (int i = 0; i < n; i++) {
        unsigned long long want = 0;
        for (int b = 0; b < 8; b++) want |= (unsigned long long)out32[i * 32 + b] << (8 * b);
        if (want != lo[i]) fails++;
    }
    printf("[selftest] contract_lo64 == low 8 bytes of contract: %d/%d ok\n", n - fails, n);
    return fails;
}

// -------------------------------------------------------------------------
// Self-test: window coverage. Run the real search kernel with prefix_len = 0,
// so every candidate "matches" and out_units becomes the exact set of units the
// window walked. It must be precisely [0, blocks*tpb*W) — no gap, no repeat, no
// overrun into the neighbouring thread's span. This is what guards the +/-i
// walk, whose failure mode is silently losing or duplicating candidates rather
// than producing wrong keys.
// -------------------------------------------------------------------------
static int check_window_coverage(int window, int blocks, int tpb) {
    const unsigned long long expect = (unsigned long long)blocks * tpb * window;
    if (expect > RESULT_CAP) { printf("[selftest] coverage W=%d skipped (too many units)\n", window); return 0; }

    uint8_t base[32]; fill_random(base, 32); clamp_scalar(base);
    uint8_t *d_base, *d_req;
    unsigned long long *d_count, *d_units;
    bignum25519 *d_gx, *d_gy, *d_gp;
    const size_t tbl = (size_t)(window / 2) + 1;
    cuda_check(cudaMalloc(&d_base, 32), "cov malloc base");
    cuda_check(cudaMalloc(&d_req, 1), "cov malloc req");
    cuda_check(cudaMalloc(&d_count, sizeof(unsigned long long)), "cov malloc count");
    cuda_check(cudaMalloc(&d_units, sizeof(unsigned long long) * RESULT_CAP), "cov malloc units");
    cuda_check(cudaMalloc(&d_gx, sizeof(bignum25519) * tbl), "cov malloc gx");
    cuda_check(cudaMalloc(&d_gy, sizeof(bignum25519) * tbl), "cov malloc gy");
    cuda_check(cudaMalloc(&d_gp, sizeof(bignum25519) * tbl), "cov malloc gp");
    cuda_check(cudaMemcpy(d_base, base, 32, cudaMemcpyHostToDevice), "cov memcpy base");
    cuda_check(cudaMemset(d_count, 0, sizeof(unsigned long long)), "cov memset");
    build_step_table_kernel<<<1, 1>>>(d_gx, d_gy, d_gp, window);
    cuda_check(cudaDeviceSynchronize(), "cov table");

    LaunchArgs la{d_base, d_gx, d_gy, d_gp, d_req, d_req, 0, d_count, d_units};
    cuda_check(launch_vanity(window, blocks, tpb, la), "cov launch");
    cuda_check(cudaDeviceSynchronize(), "cov sync");

    unsigned long long count = 0;
    cuda_check(cudaMemcpy(&count, d_count, sizeof(count), cudaMemcpyDeviceToHost), "cov count");
    std::vector<unsigned long long> units(count < RESULT_CAP ? count : RESULT_CAP);
    if (!units.empty())
        cuda_check(cudaMemcpy(units.data(), d_units, sizeof(unsigned long long) * units.size(),
                              cudaMemcpyDeviceToHost), "cov units");
    cudaFree(d_base); cudaFree(d_req); cudaFree(d_count); cudaFree(d_units);
    cudaFree(d_gx); cudaFree(d_gy); cudaFree(d_gp);

    std::sort(units.begin(), units.end());
    bool ok = (count == expect) && (units.size() == expect);
    for (unsigned long long i = 0; ok && i < expect; i++) if (units[i] != i) ok = false;
    printf("[selftest] window %5d coverage (%d x %d threads): %s (%llu/%llu units)\n",
           window, blocks, tpb, ok ? "exact" : "BROKEN", count, expect);
    return ok ? 0 : 1;
}

// -------------------------------------------------------------------------
// Self-tests: incremental identity on device + a fixed known-answer vector.
// -------------------------------------------------------------------------
static int run_selftest() {
    const int N = 512;
    std::vector<uint8_t> scalars(N * 32);
    fill_random(scalars.data(), N * 32);
    for (int i = 0; i < N; i++) clamp_scalar(scalars.data() + i * 32);

    // Force scalars[0] to a fixed known-answer input for KAT comparison.
    uint8_t kat_scalar[32];
    for (int i = 0; i < 32; i++) kat_scalar[i] = (uint8_t)(i + 1); // 01 02 03 ...
    clamp_scalar(kat_scalar);
    memcpy(scalars.data(), kat_scalar, 32);

    uint8_t *d_scalars, *d_ok, *d_pub0;
    cuda_check(cudaMalloc(&d_scalars, N * 32), "malloc");
    cuda_check(cudaMalloc(&d_ok, N), "malloc");
    cuda_check(cudaMalloc(&d_pub0, 32), "malloc");
    cuda_check(cudaMemcpy(d_scalars, scalars.data(), N * 32, cudaMemcpyHostToDevice), "memcpy");

    int tpb = 128, blk = (N + tpb - 1) / tpb;
    selftest_kernel<<<blk, tpb>>>(d_scalars, N, d_ok, d_pub0);
    cuda_check(cudaGetLastError(), "selftest launch");
    cuda_check(cudaDeviceSynchronize(), "sync");

    std::vector<uint8_t> ok(N);
    uint8_t pub0[32];
    cuda_check(cudaMemcpy(ok.data(), d_ok, N, cudaMemcpyDeviceToHost), "copy ok");
    cuda_check(cudaMemcpy(pub0, d_pub0, 32, cudaMemcpyDeviceToHost), "copy pub0");

    int fails = 0;
    for (int i = 0; i < N; i++) if (!ok[i]) fails++;
    printf("[selftest] incremental identity (s+8)*B == s*B + 8B: %d/%d ok\n", N - fails, N);

    printf("[selftest] KAT scalar    = %s\n", hex_upper(kat_scalar, 32).c_str());
    printf("[selftest] KAT pubkey    = %s\n", hex_upper(pub0, 32).c_str());
    printf("[selftest] (compare against reference scalar*B; see README)\n");

    fails += check_contract_lo64();
    // A power-of-two window, a "half" window, and >1 thread so the boundary
    // between neighbouring spans is actually exercised.
    fails += check_window_coverage(64, 2, 2);
    fails += check_window_coverage(1024, 1, 2);
    fails += check_window_coverage(1536, 1, 2);

    return fails == 0 ? 0 : 2;
}

// -------------------------------------------------------------------------
// One benchmark point: fresh context, build the step table, ~1s warm-up + ~1s
// timed measurement for a (window, blocks, tpb) config. Returns Mkeys/s, or -1
// if it does not fit / cannot launch (OOM, or the register-per-block limit at
// large tpb). Each point uses a fresh context so its numbers match a real
// standalone run (each vanity_kernel<W> otherwise piles up its own local
// reservation for the context's lifetime).
// -------------------------------------------------------------------------
static double bench_one(int w, int blocks, int tpb, int dev) {
    using clock = std::chrono::steady_clock;

    cudaDeviceReset();
    cudaSetDevice(dev);
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    const int PL = 8;                              // never-matching prefix
    std::vector<uint8_t> req(PL, 0), mask(PL, 0xFF);
    uint8_t base[32];
    fill_random(base, 32);
    clamp_scalar(base);

    uint8_t *d_base, *d_req, *d_mask;
    unsigned long long *d_count, *d_units;
    bignum25519 *d_gx, *d_gy, *d_gp;
    bool ok =
        cudaMalloc(&d_base, 32) == cudaSuccess &&
        cudaMalloc(&d_req, PL) == cudaSuccess &&
        cudaMalloc(&d_mask, PL) == cudaSuccess &&
        cudaMalloc(&d_count, sizeof(unsigned long long)) == cudaSuccess &&
        cudaMalloc(&d_units, sizeof(unsigned long long) * RESULT_CAP) == cudaSuccess &&
        cudaMalloc(&d_gx, sizeof(bignum25519) * (w / 2 + 1)) == cudaSuccess &&
        cudaMalloc(&d_gy, sizeof(bignum25519) * (w / 2 + 1)) == cudaSuccess &&
        cudaMalloc(&d_gp, sizeof(bignum25519) * (w / 2 + 1)) == cudaSuccess;
    if (ok) {
        cudaMemcpy(d_base, base, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(d_req, req.data(), PL, cudaMemcpyHostToDevice);
        cudaMemcpy(d_mask, mask.data(), PL, cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(unsigned long long));
        build_step_table_kernel<<<1, 1>>>(d_gx, d_gy, d_gp, w);
        if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); ok = false; }
    }
    if (!ok) return -1;

    LaunchArgs la{d_base, d_gx, d_gy, d_gp, d_req, d_mask, PL, d_count, d_units};
    unsigned long long per_launch = (unsigned long long)blocks * tpb * (unsigned long long)w;

    // Launches until >= target seconds elapse; returns elapsed (or -1 on error)
    // with the candidate count via out-param.
    auto run_span = [&](double target, unsigned long long &cand) -> double {
        cand = 0;
        auto t0 = clock::now();
        double el = 0;
        do {
            if (launch_vanity(w, blocks, tpb, la) != cudaSuccess) { cudaGetLastError(); return -1; }
            if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); return -1; }
            cand += per_launch;
            el = std::chrono::duration<double>(clock::now() - t0).count();
        } while (el < target);
        return el;
    };

    unsigned long long tmp = 0;
    if (run_span(1.0, tmp) < 0) return -1;         // warm-up doubles as a fit check
    unsigned long long cand = 0;
    double el = run_span(1.0, cand);
    return (el > 0) ? (double)cand / el / 1e6 : 0;
}

// -------------------------------------------------------------------------
// Quick benchmark: sweep every window at the base block size, then sweep the
// block size (tpb) at the fastest window, and report the best (window, tpb).
// ~1s warm-up + ~1s measured per point.
// -------------------------------------------------------------------------
static int run_benchmark(int blocks, int tpb) {
    int numSM = 1, maxThreadsSM = 1, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&maxThreadsSM, cudaDevAttrMaxThreadsPerMultiProcessor, dev);

    // Phase 1 — window sweep at the base block size; keep the top two fitting
    // windows for the grid sweep.
    fprintf(stderr, "Benchmarking on %d SM (blocks %d, tpb %d)...\n", numSM, blocks, tpb);
    printf("%6s  %10s  %8s  %9s\n", "window", "Mkeys/s", "loc/thr", "reserve");
    double best_mps = 0, second_mps = 0; int best_w = 0, second_w = 0;
    const int nW = (int)(sizeof(kWindows) / sizeof(kWindows[0]));
    for (int idx = nW - 1; idx >= 0; idx--) {      // ascending, small windows first
        int w = kWindows[idx];
        double mps = bench_one(w, blocks, tpb, dev);
        size_t localB = window_local_bytes(w);     // context is alive after bench_one
        size_t reserveMB = window_reserve_bytes(w, numSM, maxThreadsSM) >> 20;
        if (mps < 0)
            printf("%6d  %10s  %6zuKB  %6zuMB\n", w, "OOM/skip", localB >> 10, reserveMB);
        else {
            printf("%6d  %10.1f  %6zuKB  %6zuMB\n", w, mps, localB >> 10, reserveMB);
            if (mps > best_mps) { second_mps = best_mps; second_w = best_w; best_mps = mps; best_w = w; }
            else if (mps > second_mps) { second_mps = mps; second_w = w; }
        }
        fflush(stdout);
    }

    // Phase 2 — grid sweep over the top two windows x block size. Two windows,
    // not one, because the per-block register limit can bar a large tpb on the
    // biggest window while a slightly smaller one still allows it (the window
    // ranking is not separable from tpb). Blocks are held fixed: beyond a few
    // waves they barely matter.
    int gw = best_w, gtpb = tpb; double gmps = best_mps;
    if (best_w) {
        const int cand[2] = {best_w, second_w};
        const int ncand = second_w ? 2 : 1;
        printf("\nGrid sweep (top windows x block size):\n%8s  %6s  %10s\n",
               "window", "tpb", "Mkeys/s");
        for (int c = 0; c < ncand; c++) {
            // 384 is in the list because the kernel's register footprint puts
            // the per-block register cap between 256 and 512 on current GPUs,
            // and the resident-threads-per-SM granularity makes it a real peak.
            for (int t : {128, 256, 384, 512, 1024}) {
                double mps = bench_one(cand[c], blocks, t, dev);
                if (mps < 0) printf("%8d  %6d  %10s\n", cand[c], t, "skip");
                else {
                    printf("%8d  %6d  %10.1f\n", cand[c], t, mps);
                    if (mps > gmps) { gmps = mps; gw = cand[c]; gtpb = t; }
                }
                fflush(stdout);
            }
        }
    }

    cudaDeviceReset();
    cudaSetDevice(dev);
    printf("(reserve = worst-case local memory the driver pins = "
           "SM count x maxThreadsPerSM x loc/thr)\n");
    if (best_w)
        printf("Fastest: --window %d --tpb %d  (%.1f Mkeys/s)\n", gw, gtpb, gmps);
    return 0;
}
