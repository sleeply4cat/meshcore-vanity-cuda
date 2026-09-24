// meshcore-vanity-cuda — GPU vanity public-key search for MeshCore.
//
// Algorithm (see README): pubkey(s+8) = pubkey(s) + 8*B, so instead of a full
// fixed-base scalar multiplication per candidate we add a precomputed multiple
// of 8B to a per-thread centre point, keep only the affine y coordinate, and
// share one field inversion across a whole window of W candidates with
// Montgomery's batch-inversion trick.
//
// Every thread g has its own random base scalar and, in its k-th launch since
// that base was drawn, owns candidates s = base_g + (k*W + j)*8, j in [0,W). It
// walks that span outwards from the centre in +/-i pairs: since -Q = (-x, y),
// one table entry i*D yields both s = centre +/- 8i, and with the d-free
// addition formula the two denominators multiply to a table lookup and a
// subtraction, so a pair costs 8 field multiplications. The centre point costs
// a fixed-base multiply only when the base is drawn; after that the thread
// carries it from launch to launch.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <random>
#include <chrono>

#ifdef _WIN32
// Declared by hand rather than through <windows.h>/<bcrypt.h>, whose macros
// (min, max, near, far, ...) do not mix well with the rest of this file.
extern "C" __declspec(dllimport) long __stdcall BCryptGenRandom(void *, unsigned char *,
                                                                unsigned long, unsigned long);
#pragma comment(lib, "bcrypt.lib")
#else
#include <sys/random.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#endif

#include "ed25519.cuh"

// Window = candidates per thread per launch (batch size). It is chosen at
// runtime (--window, or auto-fit to VRAM); the kernel is templated on it so the
// per-thread buffers stay compile-time sized.
//
// Supported windows, descending. The single per-thread buffer (pref) holds one
// prefix product per +/-i PAIR, i.e. W/2 elements = W*20 bytes of local memory,
// so W=16384 = 320 KB/thread, still under the 512 KB per-thread local limit
// (numerators and denominators are recomputed, not stored). A bigger window
// amortizes the one per-window inversion over more candidates, but that is
// worth about 2% past W=2048 while the VRAM reserve grows with W (at 16384
// it is ~7.7 GB on a 16-SM GPU); pick one with --window. The 1.5x "half"
// windows (1536/3072/6144/12288) fill the gaps between powers of two so a size
// can be chosen closer to what VRAM allows; all are multiples of 512 (a warp is
// 32, a default block 256) for clean local-frame/table alignment.
static const int kWindows[] = {16384, 12288, 8192, 6144, 4096, 3072, 2048,
                               1536, 1024, 512, 256, 128, 64};

// Max matches recorded per launch. With several criteria the hit rate is their
// sum, so this needs headroom; overflow is reported rather than silently
// dropped.
#define RESULT_CAP 65536

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
// The shared step table, for Q_i = i*D (D = 8B), i in [1, W/2]:
//     x = x_i,  z = 1/y_i,  t = x_i/y_i,  s = t^2
// which is what the d-free y-only addition in the search kernel reads (see
// there). Only half a window is tabulated because entry i serves both
// s = centre + 8i and s = centre - 8i (negating a point flips only x, so it
// flips x and t and keeps z and s). The entries are identical for every thread,
// so this runs once and the search kernel just reads it (broadcast across the
// warp). Single thread; the per-entry inversion is one-time and negligible.
// -------------------------------------------------------------------------
struct StepTable { bignum25519 *x, *z, *t, *s; };

__global__ void build_step_table_kernel(StepTable tab, int window) {
    ge25519_niels D; load_step_8B(&D);
    uint8_t eight[32]; for (int k = 0; k < 32; k++) eight[k] = 0; eight[0] = 8;
    ge25519 Q; scalar_to_point(&Q, eight);   // Q = 8B = 1*D
    const int n = window / 2;
    for (int i = 1; i <= n; i++) {
        bignum25519 yz, inv;
        curve25519_mul(yz, Q.y, Q.z);
        curve25519_recip(inv, yz);                       // 1/(Y Z)
        curve25519_mul(tab.x[i], Q.x, Q.y);
        curve25519_mul(tab.x[i], tab.x[i], inv);         // x = X/Z
        curve25519_mul(tab.z[i], Q.z, Q.z);
        curve25519_mul(tab.z[i], tab.z[i], inv);         // 1/y = Z/Y
        curve25519_mul(tab.t[i], tab.x[i], tab.z[i]);    // t = x/y
        curve25519_square(tab.s[i], tab.t[i]);           // s = t^2
        if (i < n) ge25519_nielsadd2(&Q, &D);            // Q += D
    }
}

// -------------------------------------------------------------------------
// Affine coords of D = (units*8)*B, the point every thread adds to its window
// centre between launches, plus PD = xD*yD. `units` is the window W: a thread
// covers exactly W candidates per launch, so its next window starts W*8 further
// on, and the host's per-thread launch count stays in lockstep with the point.
// Runs once per configuration; single thread.
// -------------------------------------------------------------------------
__global__ void build_launch_step_kernel(bignum25519 *out3, unsigned long long units) {
    uint8_t s[32];
    for (int k = 0; k < 32; k++) s[k] = 0;
    unsigned long long v = units * 8ULL;
    for (int k = 0; k < 8; k++) s[k] = (uint8_t)(v >> (8 * k));
    ge25519 P;
    scalar_to_point(&P, s);
    ge_to_affine(out3[0], out3[1], &P);
    curve25519_mul(out3[2], out3[0], out3[1]);
}

// -------------------------------------------------------------------------
// Low 8 bytes of the compressed encoding of y, for y straight out of
// curve25519_mul, as one uint64.
//
// The filter only ever looks at the first bytes of the key and the encoding is
// little-endian, so packing all 32 bytes per candidate is waste — and so is
// most of the canonicalisation curve25519_contract performs. A multiply leaves
// every limb within its 26/25 bits except limb 1, which may carry a few bits
// over (the tail of its reduction), so the represented value is below
// 2^255 + 2^52 and bits 0..63 are just f0 + f1*2^26 + f2*2^51 (limb 3 starts
// at bit 77). The value is the canonical y unless it lies in [p, 2^255 + 2^52),
// i.e. unless y < 2^52 + 19: a chance of about 2^-203 per candidate, in which
// case a genuine match is missed, never a false one reported (the host
// re-derives every hit). --selftest checks this against the full contract.
// -------------------------------------------------------------------------
__device__ __forceinline__ static uint64_t key_lo64(const bignum25519 y) {
    return (uint64_t)y[0] + ((uint64_t)y[1] << 26) + ((uint64_t)y[2] << 51);
}

// -------------------------------------------------------------------------
// key_lo64(a*b) without the middle of the multiply.
//
// The filter reads only the low 64 bits of the final product y = a*b, and
// those depend on just a few of the ten product columns. Write the product
// columns (before any carry) as m0..m9 at bit offsets 0,26,51,77,...,230 and
// S = sum m_k 2^off_k. curve25519_mul's carry chain computes exactly
// S mod 2^255 plus 19*Q folded into limb 0, Q = floor(S / 2^255), so
//     key_lo64(a*b) = (m0 + m1 2^26 + m2 2^51 + 19 Q)  mod 2^64
// (m3 starts at bit 77). Q depends on the top: in units of 2^179, the columns
// below m7 add less than 2^37 to m9 2^51 + m8 2^25 + m7 (for the operand
// ranges below), so the nested floor over m7..m9 gives Q exactly unless that
// sum is within 2^38 below a multiple of 2^76. So 6 of the 10 columns are
// enough: 60 of the 100 partial products, and 3 carry steps instead of 11.
//
// Returns lo, and alt = true in the ~2^-38 case where Q may be one larger
// (then the true value is lo or lo + 19 and the caller tests both; the host
// re-derives every hit, so an extra candidate costs nothing but a check).
// Otherwise the result is bit-identical to key_lo64(curve25519_mul(a, b)),
// documented 2^-203 miss included. --selftest checks this on edge values and
// on random products with the search kernel's operand ranges.
//
// Operand ranges are those of curve25519_mul in the search kernel: b is the
// one scaled by 19/38, so it must be a multiply output (limbs within their 26/25
// bits, limb 1 a little over); a may be an unreduced curve25519_add or a
// partially carried curve25519_sub of multiply outputs (limbs up to ~1.5 * 2^27).
// -------------------------------------------------------------------------
__device__ __forceinline__ static uint64_t mul_key_lo64(const bignum25519 a, const bignum25519 b,
                                                        bool &alt) {
    const uint32_t s0 = a[0], s1 = a[1], s2 = a[2], s3 = a[3], s4 = a[4],
                   s5 = a[5], s6 = a[6], s7 = a[7], s8 = a[8], s9 = a[9];
    const uint32_t r0 = b[0], r1 = b[1], r2 = b[2], r3 = b[3], r4 = b[4],
                   r5 = b[5], r6 = b[6], r7 = b[7], r8 = b[8], r9 = b[9];
    // Wrapped terms (i + j >= 10) carry 19; both-odd terms carry an extra 2.
    const uint32_t r2_19 = r2 * 19, r4_19 = r4 * 19, r6_19 = r6 * 19, r8_19 = r8 * 19;
    const uint32_t r3_19 = r3 * 19, r5_19 = r5 * 19, r7_19 = r7 * 19, r9_19 = r9 * 19;
    const uint32_t r1_38 = r1 * 38, r3_38 = r3 * 38, r5_38 = r5 * 38, r7_38 = r7 * 38,
                   r9_38 = r9 * 38;
    const uint32_t r1_2 = r1 * 2, r3_2 = r3 * 2, r5_2 = r5 * 2, r7_2 = r7 * 2;
    uint32_t l0, h0, l1, h1, l2, h2, l7, h7, l8, h8, l9, h9;

    MUL64_SET(l0, h0, s0, r0);
    MUL64_ACC(l0, h0, s1, r9_38); MUL64_ACC(l0, h0, s2, r8_19); MUL64_ACC(l0, h0, s3, r7_38);
    MUL64_ACC(l0, h0, s4, r6_19); MUL64_ACC(l0, h0, s5, r5_38); MUL64_ACC(l0, h0, s6, r4_19);
    MUL64_ACC(l0, h0, s7, r3_38); MUL64_ACC(l0, h0, s8, r2_19); MUL64_ACC(l0, h0, s9, r1_38);

    MUL64_SET(l1, h1, s0, r1);    MUL64_ACC(l1, h1, s1, r0);
    MUL64_ACC(l1, h1, s2, r9_19); MUL64_ACC(l1, h1, s3, r8_19); MUL64_ACC(l1, h1, s4, r7_19);
    MUL64_ACC(l1, h1, s5, r6_19); MUL64_ACC(l1, h1, s6, r5_19); MUL64_ACC(l1, h1, s7, r4_19);
    MUL64_ACC(l1, h1, s8, r3_19); MUL64_ACC(l1, h1, s9, r2_19);

    MUL64_SET(l2, h2, s0, r2);    MUL64_ACC(l2, h2, s1, r1_2);  MUL64_ACC(l2, h2, s2, r0);
    MUL64_ACC(l2, h2, s3, r9_38); MUL64_ACC(l2, h2, s4, r8_19); MUL64_ACC(l2, h2, s5, r7_38);
    MUL64_ACC(l2, h2, s6, r6_19); MUL64_ACC(l2, h2, s7, r5_38); MUL64_ACC(l2, h2, s8, r4_19);
    MUL64_ACC(l2, h2, s9, r3_38);

    MUL64_SET(l7, h7, s0, r7);    MUL64_ACC(l7, h7, s1, r6);    MUL64_ACC(l7, h7, s2, r5);
    MUL64_ACC(l7, h7, s3, r4);    MUL64_ACC(l7, h7, s4, r3);    MUL64_ACC(l7, h7, s5, r2);
    MUL64_ACC(l7, h7, s6, r1);    MUL64_ACC(l7, h7, s7, r0);
    MUL64_ACC(l7, h7, s8, r9_19); MUL64_ACC(l7, h7, s9, r8_19);

    MUL64_SET(l8, h8, s0, r8);    MUL64_ACC(l8, h8, s1, r7_2);  MUL64_ACC(l8, h8, s2, r6);
    MUL64_ACC(l8, h8, s3, r5_2);  MUL64_ACC(l8, h8, s4, r4);    MUL64_ACC(l8, h8, s5, r3_2);
    MUL64_ACC(l8, h8, s6, r2);    MUL64_ACC(l8, h8, s7, r1_2);  MUL64_ACC(l8, h8, s8, r0);
    MUL64_ACC(l8, h8, s9, r9_38);

    MUL64_SET(l9, h9, s0, r9);    MUL64_ACC(l9, h9, s1, r8);    MUL64_ACC(l9, h9, s2, r7);
    MUL64_ACC(l9, h9, s3, r6);    MUL64_ACC(l9, h9, s4, r5);    MUL64_ACC(l9, h9, s5, r4);
    MUL64_ACC(l9, h9, s6, r3);    MUL64_ACC(l9, h9, s7, r2);    MUL64_ACC(l9, h9, s8, r1);
    MUL64_ACC(l9, h9, s9, r0);

    // Q = floor((m9 2^51 + m8 2^25 + m7) / 2^76), as nested floors.
    const uint64_t u8 = MUL64_GET(l8, h8) + (MUL64_GET(l7, h7) >> 25);
    const uint64_t u9 = MUL64_GET(l9, h9) + (u8 >> 26);
    const uint64_t q  = u9 >> 25;
    // Dropped fraction f9 2^51 + f8 2^25 + f7 can only come within 2^38 of
    // 2^76 if f9 is all ones and f8 is within 2^13 of all ones.
    alt = ((uint32_t)u9 & 0x1FFFFFFu) == 0x1FFFFFFu &&
          ((uint32_t)u8 & 0x3FFFFFFu) >= 0x3FFFFFFu - 0x2000u;
    return MUL64_GET(l0, h0) + (MUL64_GET(l1, h1) << 26) + (MUL64_GET(l2, h2) << 51) + 19 * q;
}

// -------------------------------------------------------------------------
// Stage 1 of the filter: the low 26 bits of key_lo64(a*b), from 30 of the 100
// partial products. Those bits are (m0 + 19 Q) mod 2^26 (m1 starts at bit 26),
// and here Q comes from the top two columns alone: everything from m7 down
// adds less than 2^60 to m9 2^51 + m8 2^25 (in units of 2^179), so the floor is
// exact unless that sum's fraction below 2^76 is within 2^64 of the next
// multiple — flagged as alt (~2^-12), where Q may be one larger and both
// values are tested. A criterion of 7+ hex digits rejects all but ~2^-24 of
// the candidates on these bits; the few that pass get the full 64 bits.
// Operand ranges as mul_key_lo64.
// -------------------------------------------------------------------------
__device__ __forceinline__ static uint32_t mul_key_lo26(const bignum25519 a, const bignum25519 b,
                                                        bool &alt) {
    const uint32_t s0 = a[0], s1 = a[1], s2 = a[2], s3 = a[3], s4 = a[4],
                   s5 = a[5], s6 = a[6], s7 = a[7], s8 = a[8], s9 = a[9];
    const uint32_t r0 = b[0], r1 = b[1], r2 = b[2], r3 = b[3], r4 = b[4],
                   r5 = b[5], r6 = b[6], r7 = b[7], r8 = b[8], r9 = b[9];
    const uint32_t r2_19 = r2 * 19, r4_19 = r4 * 19, r6_19 = r6 * 19, r8_19 = r8 * 19;
    const uint32_t r1_38 = r1 * 38, r3_38 = r3 * 38, r5_38 = r5 * 38, r7_38 = r7 * 38,
                   r9_38 = r9 * 38;
    const uint32_t r1_2 = r1 * 2, r3_2 = r3 * 2, r5_2 = r5 * 2, r7_2 = r7 * 2;
    uint32_t l0, h0, l8, h8, l9, h9;

    MUL64_SET(l0, h0, s0, r0);
    MUL64_ACC(l0, h0, s1, r9_38); MUL64_ACC(l0, h0, s2, r8_19); MUL64_ACC(l0, h0, s3, r7_38);
    MUL64_ACC(l0, h0, s4, r6_19); MUL64_ACC(l0, h0, s5, r5_38); MUL64_ACC(l0, h0, s6, r4_19);
    MUL64_ACC(l0, h0, s7, r3_38); MUL64_ACC(l0, h0, s8, r2_19); MUL64_ACC(l0, h0, s9, r1_38);

    MUL64_SET(l8, h8, s0, r8);    MUL64_ACC(l8, h8, s1, r7_2);  MUL64_ACC(l8, h8, s2, r6);
    MUL64_ACC(l8, h8, s3, r5_2);  MUL64_ACC(l8, h8, s4, r4);    MUL64_ACC(l8, h8, s5, r3_2);
    MUL64_ACC(l8, h8, s6, r2);    MUL64_ACC(l8, h8, s7, r1_2);  MUL64_ACC(l8, h8, s8, r0);
    MUL64_ACC(l8, h8, s9, r9_38);

    MUL64_SET(l9, h9, s0, r9);    MUL64_ACC(l9, h9, s1, r8);    MUL64_ACC(l9, h9, s2, r7);
    MUL64_ACC(l9, h9, s3, r6);    MUL64_ACC(l9, h9, s4, r5);    MUL64_ACC(l9, h9, s5, r4);
    MUL64_ACC(l9, h9, s6, r3);    MUL64_ACC(l9, h9, s7, r2);    MUL64_ACC(l9, h9, s8, r1);
    MUL64_ACC(l9, h9, s9, r0);

    // Q ~ floor((m9 2^51 + m8 2^25) / 2^76) as nested floors.
    const uint64_t u9 = MUL64_GET(l9, h9) + (MUL64_GET(l8, h8) >> 26);
    const uint64_t q = u9 >> 25;
    alt = ((uint32_t)u9 & 0x1FFFFFFu) >= 0x1FFFFFFu - 0x2000u;
    return (uint32_t)(MUL64_GET(l0, h0) + 19 * q);
}

// -------------------------------------------------------------------------
// Candidate filters. Each answers one question about the low 64 bits of the
// compressed key (pub[0] in the low byte), and the search kernel takes the
// filter as a template parameter, so every search mode is its own kernel: the
// single-prefix one compiles to exactly one masked compare. (Making it share
// code with the list scan behind a runtime branch measured ~0.7% slower.)
//
// No filter looks past the first 8 bytes. A criterion longer than that would
// need ~2^64 candidates to produce even a false positive here, and the host
// re-derives and fully re-checks every hit anyway.
// -------------------------------------------------------------------------
struct FilterOne {
    unsigned long long req8, mask8;
    __device__ __forceinline__ bool operator()(uint64_t lo) const {
        return ((lo ^ req8) & mask8) == 0;
    }
    __device__ __forceinline__ bool pre(uint32_t lo) const {
        return ((lo ^ (uint32_t)req8) & (uint32_t)mask8 & 0x3FFFFFFu) == 0;
    }
};

struct FilterList {
    const unsigned long long *__restrict__ req8;
    const unsigned long long *__restrict__ mask8;
    int n;
    __device__ __forceinline__ bool operator()(uint64_t lo) const {
        for (int j = 0; j < n; j++)
            if (((lo ^ req8[j]) & mask8[j]) == 0) return true;
        return false;
    }
    __device__ __forceinline__ bool pre(uint32_t lo) const {
        for (int j = 0; j < n; j++)
            if (((lo ^ (uint32_t)req8[j]) & (uint32_t)mask8[j] & 0x3FFFFFFu) == 0) return true;
        return false;
    }
};

// "The key starts with one unit repeated": broadcast the first unit of the key
// (its low nibble or low byte) across the word and compare under the mask of
// the digits that must match. Broadcasting rather than comparing neighbours
// matters for nibbles: the key prints each byte high nibble first, so an odd
// number of leading digits is not a contiguous run of bits in `lo`. Which unit
// happens to be broadcast does not matter — they must all be equal.
//   nibble: unit 0xF,  mult 0x1111111111111111
//   byte:   unit 0xFF, mult 0x0101010101010101
template <int K>
struct FilterRepeat {
    uint64_t unit[K], mult[K], mask[K];
    __device__ __forceinline__ bool operator()(uint64_t lo) const {
        bool hit = false;
#pragma unroll
        for (int k = 0; k < K; k++)
            hit |= ((lo ^ ((lo & unit[k]) * mult[k])) & mask[k]) == 0;
        return hit;
    }
    __device__ __forceinline__ bool pre(uint32_t lo) const {
        bool hit = false;
#pragma unroll
        for (int k = 0; k < K; k++)
            hit |= ((lo ^ ((lo & (uint32_t)unit[k]) * (uint32_t)mult[k])) & (uint32_t)mask[k]
                    & 0x3FFFFFFu) == 0;
        return hit;
    }
};

template <class A, class B>
struct FilterEither {
    A a; B b;
    __device__ __forceinline__ bool operator()(uint64_t lo) const { return a(lo) || b(lo); }
    __device__ __forceinline__ bool pre(uint32_t lo) const { return a.pre(lo) || b.pre(lo); }
};

// -------------------------------------------------------------------------
// Main search kernel — affine batched-addition walk in +/-i pairs (y only).
//
// Candidate i (scalar centre + 8i) is the point P0 + Q_i, Q_i = i*D from the
// shared table. Twisted Edwards curves (a = -1) have, besides the complete
// addition law, a "dedicated" one (Hisil, Wong, Carter, Dawson 2008) whose y
// does not involve d:
//     y3 = (x1 y1 - x2 y2) / (x1 y2 - y1 x2)
// It is exceptional only when x1/y1 = +/-x2/y2, i.e. P0 = +/-Q_i (a 2^-240
// event for a random base; see below for what that would cost). Divide top and
// bottom by y0 y_i and write w0 = 1/y0, r0 = x0/y0, E = x0 y0 (per thread) and
// x = x_i, z = 1/y_i, t = x_i/y_i, s = t^2 (the table):
//     centre + 8i:  y = w0 (E z - x) / (r0 - t)
//     centre - 8i:  y = w0 (E z + x) / (r0 + t)     (-Q_i = (-x_i, y_i))
// The two denominators share one batch-inversion slot, and their product is
//     (r0 - t)(r0 + t) = R - s,   R = r0^2
// — a subtraction, no multiply at all. Folding w0 into the one inversion, the
// split back into the two inverses (one multiply each) directly yields
// w0/(r0 -/+ t). Per PAIR that is 8 multiplies: 1 forward (prefix product),
// 2 backward (prefix inverse, strip), E*z, 2 splits and the 2 final products —
// against 11 multiplies + 2 squarings for the complete formula it replaces.
// The final products only feed the filter, so they are never formed: 30 of
// their 100 partial products decide nearly every candidate (mul_key_lo26), and
// the rare survivor gets 60 more (mul_key_lo64).
//
// If a denominator ever were zero, the whole window's inversion would come out
// 0 and that thread would find nothing in it (no wrong key can come out: the
// host re-derives every hit). At 2^-240 per candidate this is not a concern.
//
// Occupancy tuning: forcing the register count far down (maxrregcount=128, to
// unlock tpb=512) was benchmarked and loses — the kernel is ALU-bound, high ILP
// saturates the integer units even at low occupancy, so the extra resident
// warps do not pay for the spills. The cap does matter at the margin, though: a
// block may use 65536 registers, allocated per warp in units of 256, so a
// 12-warp block (tpb=384) needs <= 168 registers per thread. Left alone the
// kernel lands just above that and tpb=384 stops launching. __launch_bounds__
// pins it to the useful side of that cliff, and does it in the source, so every
// build path (Makefile, `make release`, the Windows CI nvcc line) inherits it.
#define VANITY_MAX_TPB 384

// Per-thread state carried between launches: the window-centre point and 1/y0.
struct WalkState { bignum25519 *x, *y, *w; };

template <int W, class Filter>
__global__ __launch_bounds__(VANITY_MAX_TPB) void vanity_kernel(
        const uint8_t *__restrict__ bases,
        StepTable tab,
        unsigned long long *__restrict__ out_count,
        unsigned long long *__restrict__ out_units,
        const bignum25519 *__restrict__ dstep,
        WalkState st,
        uint8_t *__restrict__ fresh,
        Filter match) {
    const int H = W / 2;
    const bignum25519 *__restrict__ tx = tab.x;
    const bignum25519 *__restrict__ tz = tab.z;
    const bignum25519 *__restrict__ tt = tab.t;
    const bignum25519 *__restrict__ ts = tab.s;
    const unsigned long long gid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    // Recorded hits are numbered gid*W + j, j in [0,W) the candidate's place in
    // this thread's window, so the host can tell threads apart; the scalar is
    // base_gid + (k*W + j)*8 for the thread's k-th launch on that base.
    const unsigned long long centre_unit = gid * (unsigned long long)W + (unsigned long long)H;

    // The window-centre affine point and 1/y0. When the thread has a fresh
    // base (first launch, or the host re-drew it after a hit) it costs a
    // fixed-base comb plus one inversion (~600 field multiplies); otherwise the
    // thread picks up what it left behind, advanced by exactly one window (see
    // the tail of this function).
    bignum25519 x0, y0, w0;
    if (fresh[gid]) {
        uint8_t s0[32];
        scalar_add_u64(s0, bases + gid * 32, (unsigned long long)H * 8ULL);
        clamp_scalar(s0);
        ge25519 P;
        scalar_to_point(&P, s0);
        bignum25519 yz, inv;                           // one inversion for all three
        curve25519_mul(yz, P.y, P.z);
        curve25519_recip(inv, yz);                     // 1/(Y Z)
        curve25519_mul(x0, P.x, P.y); curve25519_mul(x0, x0, inv);   // X/Z
        curve25519_mul(y0, P.y, P.y); curve25519_mul(y0, y0, inv);   // Y/Z
        curve25519_mul(w0, P.z, P.z); curve25519_mul(w0, w0, inv);   // Z/Y
        fresh[gid] = 0;
    } else {
        curve25519_copy(x0, st.x[gid]);
        curve25519_copy(y0, st.y[gid]);
        curve25519_copy(w0, st.w[gid]);
    }

    // Per-thread constants: r0 = x0/y0, E = x0 y0, R = r0^2.
    bignum25519 r0, E, R;
    curve25519_mul(r0, x0, w0);
    curve25519_mul(E, x0, y0);
    curve25519_square(R, r0);

    // Every y reaching the filter is a curve25519_mul output, as key_lo64 needs.
    auto record = [&](unsigned long long unit) {
        unsigned long long slot = atomicAdd(out_count, 1ULL);
        if (slot < RESULT_CAP) out_units[slot] = unit;
    };
    auto check_and_record = [&](const bignum25519 y, unsigned long long unit) {
        if (match(key_lo64(y))) record(unit);
    };
    // The same for y = n * d, without forming y (see mul_key_lo64).
    auto check_product = [&](const bignum25519 n, const bignum25519 d, unsigned long long unit) {
#ifdef VANITY_FULL_FINAL_MUL
        bignum25519 y;
        curve25519_mul(y, n, d);
        check_and_record(y, unit);
#else
        bool alt;
        const uint32_t lo26 = mul_key_lo26(n, d, alt);
        if (match.pre(lo26) || (alt && match.pre(lo26 + 19))) {
            const uint64_t lo = mul_key_lo64(n, d, alt);
            if (match(lo) || (alt && match(lo + 19))) record(unit);
        }
#endif
    };

    // The centre candidate needs no table entry and no inversion: y = y0.
    check_and_record(y0, centre_unit);

    // The step to the next window centre, P0 + D with D = (xD, yD, xD yD) =
    // dstep, uses the same d-free formulas:
    //     x' = (x0 y0 + xD yD) / (y0 yD - x0 xD)      = (E + PD) / Dx
    //     y' = (x0 y0 - xD yD) / (x0 yD - y0 xD)      =  Ny / Dy
    //     w' = 1/y'                                    =  Dy / Ny
    // Its three denominators seed the batch-inversion chain (as the empty
    // prefix product, so they need no stored slot), and come back out of the
    // accumulator at the end. Recomputed there rather than kept live.
    auto step_dens = [&](bignum25519 Dx, bignum25519 Dy, bignum25519 Ny) {
        bignum25519 u, v;
        curve25519_mul(u, y0, dstep[1]); curve25519_mul(v, x0, dstep[0]);
        curve25519_sub(Dx, u, v);
        curve25519_mul(u, x0, dstep[1]); curve25519_mul(v, y0, dstep[0]);
        curve25519_sub(Dy, u, v);
        curve25519_sub(Ny, E, dstep[2]);
    };

    // Pairs i = 1..H: one slot per pair holding R - s_i. Only the prefix
    // products are stored (in local memory); the slot value itself is a
    // subtraction and is simply redone in the backward pass. Additions and
    // subtractions stay unreduced (curve25519_add/sub) wherever the result
    // only feeds a multiply, which accepts the extra headroom.
    bignum25519 pref[W / 2];
    bignum25519 acc;
    {
        bignum25519 Dx, Dy, Ny;
        step_dens(Dx, Dy, Ny);
        curve25519_mul(acc, Dx, Dy);
        curve25519_mul(acc, acc, Ny);
    }
    for (int i = 1; i <= H; i++) {
        bignum25519 prod;
        curve25519_sub(prod, R, ts[i]);                // (r0 - t)(r0 + t) = R - s
        curve25519_copy(pref[i - 1], acc);             // prefix product
        curve25519_mul(acc, acc, prod);
    }
    curve25519_recip(acc, acc);                        // 1 / (step * prod(R - s))
    curve25519_mul(acc, acc, w0);                      // ... times w0, for every slot

    for (int i = H; i >= 1; i--) {
        bignum25519 invprod, prod, a, d, n;
        curve25519_mul(invprod, acc, pref[i - 1]);     // w0 / ((r0 - t)(r0 + t))
        curve25519_sub(prod, R, ts[i]);
        curve25519_mul(acc, acc, prod);                // strip this pair
        curve25519_mul(a, E, tz[i]);                   // E z

        // centre - 8i : y = w0 (E z + x) / (r0 + t),  w0/(r0 + t) = (r0 - t) * invprod
        curve25519_sub(d, r0, tt[i]);
        curve25519_mul(d, d, invprod);
        curve25519_add(n, a, tx[i]);
        check_product(n, d, centre_unit - (unsigned long long)i);

        // centre + 8i : y = w0 (E z - x) / (r0 - t),  w0/(r0 - t) = (r0 + t) * invprod.
        // i == H would land on the next window's first unit, so skip it — the
        // span stays exactly W units wide and windows never overlap.
        if (i < H) {
            curve25519_add(d, r0, tt[i]);
            curve25519_mul(d, d, invprod);
            curve25519_sub(n, a, tx[i]);
            check_product(n, d, centre_unit + (unsigned long long)i);
        }
    }

    // Every pair has been stripped, so acc = w0 / (Dx Dy Ny); times y0 it is
    // the plain inverse of the seed, and each denominator's inverse is the
    // product of the other two times that.
    {
        bignum25519 Dx, Dy, Ny, inv, t, u;
        step_dens(Dx, Dy, Ny);
        curve25519_mul(inv, acc, y0);
        curve25519_mul(t, Dy, Ny); curve25519_mul(t, t, inv);    // 1/Dx
        curve25519_add(u, E, dstep[2]);
        curve25519_mul(u, u, t);
        curve25519_copy(st.x[gid], u);                            // x'
        curve25519_mul(t, Dx, Ny); curve25519_mul(t, t, inv);    // 1/Dy
        curve25519_mul(u, Ny, t);
        curve25519_copy(st.y[gid], u);                            // y'
        curve25519_mul(t, Dx, Dy); curve25519_mul(t, t, inv);    // 1/Ny
        curve25519_mul(u, Dy, t);
        curve25519_copy(st.w[gid], u);                            // w' = 1/y'
    }
}

// -------------------------------------------------------------------------
// Selftest helper: multiply, then pack the product both ways, so the host can
// check the filter's shortcut against donna's full 32-byte contract.
// -------------------------------------------------------------------------
// Selftest helper: the partial final products against the full multiply on
// random operands shaped like the search kernel's — a is an unreduced add or a
// partially carried sub of multiply outputs, b a multiply output. Counts:
// [0] tested, [1] lo64 mismatches, [2] lo26 mismatches, [3] lo26 flagged
// ambiguous, [4] of those, the ones that really needed Q + 1.
__device__ static uint32_t test_rng(uint64_t &s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(s >> 32);
}
__device__ static void test_fe(bignum25519 f, uint64_t &s) {
    for (int k = 0; k < 10; k++) f[k] = test_rng(s) & ((k & 1) ? 0x1ffffffu : 0x3ffffffu);
}
__global__ void partial_mul_kernel(unsigned long long *cnt, int per, unsigned long long seed) {
    uint64_t s = seed ^ (0x9E3779B97F4A7C15ULL * (blockIdx.x * blockDim.x + threadIdx.x + 1));
    unsigned long long c[5] = {0, 0, 0, 0, 0};
    for (int it = 0; it < per; it++) {
        bignum25519 u, v, w, x, d, n, y;
        test_fe(u, s); test_fe(v, s); test_fe(w, s); test_fe(x, s);
        curve25519_mul(d, u, v);
        curve25519_mul(w, w, x);
        test_fe(x, s);
        if (it & 1) curve25519_add(n, w, x);
        else        curve25519_sub(n, w, x);
        curve25519_mul(y, n, d);
        const uint64_t full = key_lo64(y);
        bool alt;
        const uint64_t lo = mul_key_lo64(n, d, alt);
        c[0]++;
        if (!(lo == full || (alt && lo + 19 == full))) c[1]++;
        const uint32_t p26 = mul_key_lo26(n, d, alt) & 0x3FFFFFFu, f26 = (uint32_t)full & 0x3FFFFFFu;
        if (alt) { c[3]++; if (p26 != f26) c[4]++; }
        if (!(p26 == f26 || (alt && ((p26 + 19) & 0x3FFFFFFu) == f26))) c[2]++;
    }
    for (int k = 0; k < 5; k++) atomicAdd(&cnt[k], c[k]);
}

__global__ void mul_lo64_kernel(const bignum25519 *__restrict__ a, const bignum25519 *__restrict__ b,
                                int n, uint8_t *__restrict__ out32,
                                unsigned long long *__restrict__ out_lo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bignum25519 m;
    curve25519_mul(m, a[i], b[i]);
    curve25519_contract(out32 + i * 32, m);
    out_lo[i] = key_lo64(m);
    // The search kernel's shortcuts must agree with the full multiply: the same
    // value, or (flagged) 19 less. Report a disagreement as an impossible value.
    const uint64_t full = out_lo[i];
    bool alt;
    const uint64_t part = mul_key_lo64(a[i], b[i], alt);
    if (!(part == full || (alt && part + 19 == full))) out_lo[i] = ~full;
    const uint32_t p26 = mul_key_lo26(a[i], b[i], alt) & 0x3FFFFFFu, f26 = (uint32_t)full & 0x3FFFFFFu;
    if (!(p26 == f26 || (alt && ((p26 + 19) & 0x3FFFFFFu) == f26))) out_lo[i] = ~full;
}

// Host-callable single-key pack: full compressed pubkey (incl. parity) for a
// given clamped scalar. Used to verify/display hits.
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

// Key material comes straight from the OS CSPRNG. The private key is the base
// scalar plus a known small offset, and the signing half seeds every signature
// nonce, so both carry exactly the entropy of this source: a seeded userspace
// PRNG would cap them at the size of its seed.
static void fill_random(uint8_t *p, size_t n) {
#ifdef _WIN32
    while (n) {
        unsigned long chunk = n > (1u << 30) ? (1u << 30) : (unsigned long)n;
        // 2 = BCRYPT_USE_SYSTEM_PREFERRED_RNG
        if (BCryptGenRandom(nullptr, p, chunk, 2) != 0) {
            fprintf(stderr, "BCryptGenRandom failed\n");
            exit(1);
        }
        p += chunk; n -= chunk;
    }
#else
    while (n) {
        ssize_t r = getrandom(p, n, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0 && errno == ENOSYS) {                // pre-3.17 kernel
            int fd = open("/dev/urandom", O_RDONLY);
            while (fd >= 0 && n) {
                ssize_t k = read(fd, p, n);
                if (k < 0 && errno == EINTR) continue;
                if (k <= 0) break;
                p += k; n -= (size_t)k;
            }
            if (fd >= 0) close(fd);
            if (n == 0) return;
        }
        if (r <= 0) {
            fprintf(stderr, "Cannot read the system random source\n");
            exit(1);
        }
        p += r; n -= (size_t)r;
    }
#endif
}

// n independent random clamped base scalars, 32 bytes each.
static void random_bases(uint8_t *p, size_t n) {
    fill_random(p, n * 32);
    for (size_t i = 0; i < n; i++) clamp_scalar(p + i * 32);
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

// Parse a hex prefix into (req, mask) byte vectors, like the reference
// build_matcher_from_hex. The caller has validated the characters. Returns
// prefix_len in bytes.
static int build_matcher(const std::string &hex, std::vector<uint8_t> &req,
                         std::vector<uint8_t> &mask) {
    int full_bytes = (int)(hex.size() + 1) / 2;
    req.assign(full_bytes, 0);
    mask.assign(full_bytes, 0);
    for (size_t i = 0; i < hex.size(); i++) {
        int bi = (int)i / 2, v = hexval(hex[i]);
        if (i % 2 == 0) { req[bi] |= v << 4; mask[bi] |= 0xF0; }
        else            { req[bi] |= v;      mask[bi] |= 0x0F; }
    }
    return full_bytes;
}

// The first (up to) 8 bytes of a prefix as one masked 64-bit comparison.
static void pack_prefix(const std::string &hex, unsigned long long &req8,
                        unsigned long long &mask8) {
    std::vector<uint8_t> req, mask;
    int len = build_matcher(hex, req, mask);
    req8 = mask8 = 0;
    for (int i = 0; i < 8 && i < len; i++) {
        req8  |= (unsigned long long)req[i]  << (8 * i);
        mask8 |= (unsigned long long)mask[i] << (8 * i);
    }
}

static std::string hex_upper(const uint8_t *p, int n) {
    static const char *H = "0123456789ABCDEF";
    std::string s;
    s.reserve(n * 2);
    for (int i = 0; i < n; i++) { s += H[p[i] >> 4]; s += H[p[i] & 0xf]; }
    return s;
}

static std::string upper(std::string s) {
    for (char &c : s) c = (char)toupper((unsigned char)c);
    return s;
}

// Minimal base-10 parser for CLI integers. Deliberately avoids atoi/atol/strtol:
// with a recent GCC/glibc those get redirected to __isoc23_strtol (GLIBC_2.38),
// which would break the binary on older distros (e.g. Ubuntu 22.04 / glibc 2.35).
// This keeps the highest required glibc symbol at 2.34. Anything but plain
// digits is rejected, so a typo cannot silently become 0.
static long parse_num(const char *s, const char *opt) {
    long v = 0;
    const char *p = s;
    for (; *p >= '0' && *p <= '9'; p++) {
        v = v * 10 + (*p - '0');
        if (v > 1000000000L) break;
    }
    if (p == s || *p) {
        fprintf(stderr, "%s expects a non-negative number, got '%s'\n", opt, s);
        exit(1);
    }
    return v;
}

// -------------------------------------------------------------------------
// Search criteria: hex prefixes and repeat rules. A key counts when it meets
// any of them. Everything here works on the key's hex digits ("nibbles"), in
// the order the key prints.
// -------------------------------------------------------------------------

// The first `n` hex digits are one `u`-digit unit repeated: u = 1 is AAAA...,
// u = 2 is ABABAB... (the same byte n/2 times).
struct RepeatRule { int n, u; };

struct Criteria {
    std::vector<std::string> prefixes;     // lowercase hex
    std::vector<RepeatRule> rules;         // at most one per unit size
};

static void key_nibbles(const uint8_t pub[32], uint8_t nib[64]) {
    for (int i = 0; i < 32; i++) { nib[2 * i] = pub[i] >> 4; nib[2 * i + 1] = pub[i] & 0xF; }
}

static std::vector<uint8_t> str_nibbles(const std::string &hex) {
    std::vector<uint8_t> v;
    for (char c : hex) v.push_back((uint8_t)hexval(c));
    return v;
}

static bool is_periodic(const uint8_t *nib, int len, int u) {
    for (int i = u; i < len; i++) if (nib[i] != nib[i - u]) return false;
    return true;
}

// Length of the leading run of the key that repeats its first u digits.
static int periodic_len(const uint8_t *nib, int len, int u) {
    int i = u;
    while (i < len && nib[i] == nib[i - u]) i++;
    return i;
}

static bool nib_has_prefix(const uint8_t *nib, const std::vector<uint8_t> &p) {
    for (size_t i = 0; i < p.size(); i++) if (nib[i] != p[i]) return false;
    return true;
}

// Which criterion a key (given as nibbles) meets: the index of a prefix, or
// prefixes.size() + the index of a rule, or -1. Prefixes are tried first.
static int match_nibbles(const Criteria &c, const uint8_t *nib, int len) {
    for (size_t j = 0; j < c.prefixes.size(); j++)
        if ((int)c.prefixes[j].size() <= len && nib_has_prefix(nib, str_nibbles(c.prefixes[j])))
            return (int)j;
    for (size_t r = 0; r < c.rules.size(); r++)
        if (c.rules[r].n <= len && is_periodic(nib, c.rules[r].n, c.rules[r].u))
            return (int)(c.prefixes.size() + r);
    return -1;
}

// Rule a => rule b (every key meeting a meets b).
static bool rule_implies(RepeatRule a, RepeatRule b) {
    return b.u % a.u == 0 && a.n >= b.n;
}

// Prefix p => rule r.
static bool prefix_implies(const std::string &p, RepeatRule r) {
    std::vector<uint8_t> v = str_nibbles(p);
    return (int)v.size() >= r.n && is_periodic(v.data(), r.n, r.u);
}

// Drop criteria that another one already covers: exact and nested duplicate
// prefixes ("abcd" when "ab" is there), prefixes a rule covers, and a rule
// another rule covers. This costs nothing in the result — the union is the
// same — but it saves filter work, and it makes the remaining prefixes
// pairwise disjoint, which the probability below relies on.
static void normalize(Criteria &c, bool verbose) {
    for (std::string &p : c.prefixes)
        for (char &ch : p) ch = (char)tolower((unsigned char)ch);
    std::stable_sort(c.prefixes.begin(), c.prefixes.end(),
                     [](const std::string &a, const std::string &b) { return a.size() < b.size(); });

    std::vector<RepeatRule> rules;
    for (size_t i = 0; i < c.rules.size(); i++) {
        bool covered = false;
        for (size_t k = 0; k < c.rules.size() && !covered; k++)
            if (k != i && rule_implies(c.rules[i], c.rules[k]) &&
                !(rule_implies(c.rules[k], c.rules[i]) && k > i))
                covered = true;
        if (covered) {
            if (verbose)
                fprintf(stderr, "Note: --repeat-nibble %d is already covered by --repeat-byte; dropped.\n",
                        c.rules[i].n);
        } else rules.push_back(c.rules[i]);
    }
    c.rules = rules;

    std::vector<std::string> kept;
    for (const std::string &p : c.prefixes) {
        const char *why = nullptr; std::string by;
        for (const std::string &q : kept)
            if (p.compare(0, q.size(), q) == 0) { why = "prefix"; by = q; break; }
        for (size_t r = 0; !why && r < c.rules.size(); r++)
            if (prefix_implies(p, c.rules[r])) why = c.rules[r].u == 1 ? "--repeat-nibble" : "--repeat-byte";
        if (!why) { kept.push_back(p); continue; }
        if (verbose) {
            if (by.empty()) fprintf(stderr, "Note: prefix %s is already covered by %s; dropped.\n",
                                    upper(p).c_str(), why);
            else fprintf(stderr, "Note: prefix %s is already covered by prefix %s; dropped.\n",
                         upper(p).c_str(), upper(by).c_str());
        }
    }
    c.prefixes = kept;
}

static double p_digits(int k) { return ldexp(1.0, -4 * k); }   // 16^-k

static double p_rule(RepeatRule r) { return p_digits(r.n - r.u); }

// P(key starts with prefix p AND meets rule r).
static double p_prefix_and_rule(const std::string &p, RepeatRule r) {
    std::vector<uint8_t> v = str_nibbles(p);
    const int L = (int)v.size();
    if (!is_periodic(v.data(), std::min(L, r.n), r.u)) return 0;
    if (L >= r.n) return p_digits(L);                  // p implies r
    // The prefix fixes min(L, u) digits of the unit; the rest are free.
    return p_digits(r.n - (r.u - std::min(L, r.u)));
}

// Per-candidate probability of meeting any criterion, exact by
// inclusion-exclusion. Needs normalize() first: prefixes are then pairwise
// disjoint. Two rules (nibble n1, byte n2 digits, n1 < n2 after normalize)
// intersect in "n2 identical digits", itself a nibble rule.
static double hit_probability(const Criteria &c) {
    std::vector<RepeatRule> both;
    if (c.rules.size() == 2)
        both.push_back({std::max(c.rules[0].n, c.rules[1].n), std::min(c.rules[0].u, c.rules[1].u)});
    double p = 0;
    for (RepeatRule r : c.rules) p += p_rule(r);
    for (RepeatRule r : both) p -= p_rule(r);
    for (const std::string &s : c.prefixes) {
        p += p_digits((int)s.size());
        for (RepeatRule r : c.rules) p -= p_prefix_and_rule(s, r);
        for (RepeatRule r : both) p += p_prefix_and_rule(s, r);
    }
    return p;
}

// Low-64-bit mask of the first min(n, 16) hex digits (each byte high nibble
// first, pub[0] in the low byte).
static unsigned long long digit_mask(int n) {
    unsigned long long m = 0;
    for (int i = 0; i < n && i < 16; i++) m |= 0xFULL << (8 * (i / 2) + ((i & 1) ? 0 : 4));
    return m;
}

template <int K>
static FilterRepeat<K> make_repeat(const std::vector<RepeatRule> &rules) {
    FilterRepeat<K> f;
    for (int k = 0; k < K; k++) {
        f.unit[k] = rules[k].u == 1 ? 0xFULL : 0xFFULL;
        f.mult[k] = rules[k].u == 1 ? 0x1111111111111111ULL : 0x0101010101010101ULL;
        f.mask[k] = digit_mask(rules[k].n);
    }
    return f;
}

// A prefix list in device memory, for FilterList.
struct DevList {
    unsigned long long *req8 = nullptr, *mask8 = nullptr;
    int n = 0;
    explicit DevList(const std::vector<std::string> &prefixes) : n((int)prefixes.size()) {
        if (!n) return;
        std::vector<unsigned long long> r(n), m(n);
        for (int j = 0; j < n; j++) pack_prefix(prefixes[j], r[j], m[j]);
        cuda_check(cudaMalloc(&req8, sizeof(unsigned long long) * n), "malloc req8");
        cuda_check(cudaMalloc(&mask8, sizeof(unsigned long long) * n), "malloc mask8");
        cuda_check(cudaMemcpy(req8, r.data(), sizeof(unsigned long long) * n,
                              cudaMemcpyHostToDevice), "memcpy req8");
        cuda_check(cudaMemcpy(mask8, m.data(), sizeof(unsigned long long) * n,
                              cudaMemcpyHostToDevice), "memcpy mask8");
    }
    ~DevList() { cudaFree(req8); cudaFree(mask8); }
    DevList(const DevList &) = delete;
    DevList &operator=(const DevList &) = delete;
    FilterList filter() const { return FilterList{req8, mask8, n}; }
};

// Pack one clamped scalar -> full compressed pubkey (device round-trip).
static void pack_one(const uint8_t scalar[32], uint8_t pub[32],
                    uint8_t *d_scalar, uint8_t *d_pub) {
    cuda_check(cudaMemcpy(d_scalar, scalar, 32, cudaMemcpyHostToDevice), "memcpy scalar");
    pack_one_kernel<<<1, 1>>>(d_scalar, d_pub);
    cuda_check(cudaGetLastError(), "pack_one launch");
    cuda_check(cudaMemcpy(pub, d_pub, 32, cudaMemcpyDeviceToHost), "memcpy pub");
}

// Kernel launch args (everything but the filter), bundled so the window
// dispatch stays readable.
struct LaunchArgs {
    const uint8_t *bases;                // per-thread base scalars, 32 bytes each
    StepTable tab;                       // shared step table
    unsigned long long *count, *units;
    const bignum25519 *dstep;            // affine (xD, yD, xD*yD) of the launch step
    WalkState st;                        // persistent per-thread centre and 1/y
    uint8_t *fresh;                      // per thread: 1 = seed from its base
};

// Device allocations for the step table (entries 1..window/2) and the walk
// state. Return false on failure, leaving what was allocated to the free call.
static bool alloc_table(StepTable &t, int window) {
    const size_t b = sizeof(bignum25519) * (size_t)(window / 2 + 1);
    t = StepTable{nullptr, nullptr, nullptr, nullptr};
    return cudaMalloc(&t.x, b) == cudaSuccess && cudaMalloc(&t.z, b) == cudaSuccess &&
           cudaMalloc(&t.t, b) == cudaSuccess && cudaMalloc(&t.s, b) == cudaSuccess;
}
static void free_table(StepTable &t) { cudaFree(t.x); cudaFree(t.z); cudaFree(t.t); cudaFree(t.s); }

static bool alloc_state(WalkState &w, unsigned long long threads) {
    const size_t b = sizeof(bignum25519) * threads;
    w = WalkState{nullptr, nullptr, nullptr};
    return cudaMalloc(&w.x, b) == cudaSuccess && cudaMalloc(&w.y, b) == cudaSuccess &&
           cudaMalloc(&w.w, b) == cudaSuccess;
}
static void free_state(WalkState &w) { cudaFree(w.x); cudaFree(w.y); cudaFree(w.w); }

// Launch the vanity kernel instantiation for a runtime window and filter.
// Returns the launch error (e.g. cudaErrorMemoryAllocation if the local frame
// won't fit).
template <class F>
static cudaError_t launch_vanity(int window, int blocks, int tpb, const LaunchArgs &a, const F &f) {
#define LV(W) case W: vanity_kernel<W, F><<<blocks, tpb>>>(a.bases, a.tab, a.count, a.units, \
                  a.dstep, a.st, a.fresh, f); break;
    switch (window) {
        LV(16384) LV(12288) LV(8192) LV(6144) LV(4096) LV(3072) LV(2048)
        LV(1536) LV(1024) LV(512) LV(256) LV(128) LV(64)
        default: return cudaErrorInvalidValue;
    }
#undef LV
    return cudaGetLastError();
}

// All filters give the kernel the same local frame (the pref buffer), so the
// runtime introspection helpers (attributes, occupancy) use the single-prefix
// one.
static const void *vanity_kernel_ptr(int window) {
#define KP(W) case W: return (const void *)vanity_kernel<W, FilterOne>;
    switch (window) {
        KP(16384) KP(12288) KP(8192) KP(6144) KP(4096) KP(3072) KP(2048)
        KP(1536) KP(1024) KP(512) KP(256) KP(128)
        default: return (const void *)vanity_kernel<64, FilterOne>;
    }
#undef KP
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

// Grid size. Every thread's work is identical, so blocks finish in lockstep
// "waves" of (SM count x blocks resident per SM), and a grid that is not a
// whole number of waves leaves most SMs idle during the last one: 512 blocks on
// a 170-SM GPU is 3.01 waves, i.e. four waves of time for three of work. So the
// default grid is always whole waves. Many of them, too: with several blocks per
// SM, blocks drift out of phase over the waves, so the serial stretches of a
// window (the inversion) overlap other blocks' multiply-heavy loops instead of
// all SMs' warps stalling on them at once — worth ~14% on an RTX 5060 Ti from 5
// to 40 waves, while one block per SM (tpb 384) stays flat. Registers, not
// shared memory, bound residency (168 per thread: 12 warps per SM), so blocks
// per SM comes from the occupancy calculator for the actual kernel.
#define DEFAULT_WAVES 32
#define STR_(x) #x
#define STR(x) STR_(x)
#define DEFAULT_TPB 128

static int blocks_per_sm(int window, int tpb) {
    int bps = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, vanity_kernel_ptr(window), tpb, 0)
            != cudaSuccess || bps < 1) {
        cudaGetLastError();
        bps = 1;             // not resident at all: the launch itself will say so
    }
    return bps;
}

static int wave_blocks(int window, int tpb, int waves) {
    int dev = 0, numSM = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev);
    return numSM * blocks_per_sm(window, tpb) * waves;
}

// Largest supported window whose local-memory reserve fits free VRAM with a
// margin, but not larger than AUTO_WINDOW_CAP.
//
// The cap exists because the persistent walk removed the reason to go big. The
// per-window fixed-base multiply and its inversion used to cost ~600 field
// multiplies per thread per launch, which only a large W could amortise; now a
// thread carries its centre across launches and pays ~12. What is left that
// still scales with W is the single Montgomery inversion, ~180 multiplies per
// window against ~4 per candidate — so going past 2048 can buy at most
// 180/2048/4 ≈ 2%, while the memory reserve grows in proportion to W. That
// bound is arithmetic, not hardware-specific. --window and --benchmark are
// still there for anyone who wants to chase the last percent.
#define AUTO_WINDOW_CAP 2048
static int auto_window() {
    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);
    int numSM = 1, maxThreadsSM = 1024, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&maxThreadsSM, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    for (int w : kWindows)
        if (w <= AUTO_WINDOW_CAP &&
            (double)window_reserve_bytes(w, numSM, maxThreadsSM) < 0.85 * (double)freeB)
            return w;
    return kWindows[sizeof(kWindows) / sizeof(kWindows[0]) - 1];  // smallest
}

// Largest supported window <= req (the smallest one if req is below them all).
static int nearest_window(int req) {
    for (int w : kWindows) if (w <= req) return w;
    return kWindows[sizeof(kWindows) / sizeof(kWindows[0]) - 1];
}

static int run_selftest();
static int run_benchmark();

int main(int argc, char **argv) {
    Criteria crit;
    long limit = 1;
    int blocks = 0;              // 0 = whole waves (DEFAULT_WAVES)
    int tpb = DEFAULT_TPB;
    bool grid_given = false;
    int device = 0;
    int window = 0;              // 0 = auto-fit to VRAM
    bool progress = true;
    bool selftest = false;
    bool benchmark = false;

    auto set_rule = [&](RepeatRule r) {
        for (RepeatRule &q : crit.rules) if (q.u == r.u) { q = r; return; }
        crit.rules.push_back(r);
    };

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char *n) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", n); exit(1); }
            return argv[++i];
        };
        if (a == "--selftest") selftest = true;
        else if (a == "--benchmark" || a == "--bench") benchmark = true;
        else if (a == "-l" || a == "--limit") limit = parse_num(need("--limit"), "--limit");
        else if (a == "--blocks") { blocks = (int)parse_num(need("--blocks"), "--blocks"); grid_given = true;
                                    if (blocks < 1) { fprintf(stderr, "--blocks must be at least 1.\n"); return 1; } }
        else if (a == "--tpb") { tpb = (int)parse_num(need("--tpb"), "--tpb"); grid_given = true; }
        else if (a == "-d" || a == "--device") device = (int)parse_num(need("--device"), "--device");
        else if (a == "-w" || a == "--window") window = (int)parse_num(need("--window"), "--window");
        else if (a == "--repeat-nibble") {
            long n = parse_num(need("--repeat-nibble"), "--repeat-nibble");
            if (n < 2 || n > 64) { fprintf(stderr, "--repeat-nibble takes 2..64 digits\n"); return 1; }
            set_rule({(int)n, 1});
        }
        else if (a == "--repeat-byte") {
            long n = parse_num(need("--repeat-byte"), "--repeat-byte");
            if (n < 2 || n > 32) { fprintf(stderr, "--repeat-byte takes 2..32 bytes\n"); return 1; }
            set_rule({(int)(2 * n), 2});
        }
        else if (a == "--no-progress") progress = false;
        else if (a == "-h" || a == "--help") {
            printf("Usage: %s [HEX_PREFIX ...] [options]\n"
                   "  A key counts when it meets ANY of the criteria below; several are\n"
                   "  searched in one pass, which divides the expected time accordingly.\n"
                   "  HEX_PREFIX            key starts with these hex digits\n"
                   "      --repeat-nibble N key starts with N+ identical hex digits (AAAA..)\n"
                   "      --repeat-byte N   key starts with N+ identical bytes (ABABAB..)\n"
                   "  -l, --limit N     stop after N matches total (0 = infinite) [1]\n"
                   "  -w, --window N    batch/thr: 64..16384 incl. 1536/3072/6144/12288 [auto VRAM]\n"
                   "                    past ~2048 at most ~2%% faster, but much more GPU memory\n"
                   "      --blocks N    CUDA blocks [auto: 32 whole waves for this GPU]\n"
                   "      --tpb N       threads per block, 32..384 [128]\n"
                   "  -d, --device I    CUDA device index [0]\n"
                   "      --no-progress suppress progress output\n"
                   "      --selftest    run correctness self-tests and exit\n"
                   "      --benchmark   sweep window + block size, print the fastest, and exit\n",
                   argv[0]);
            return 0;
        }
        else if (!a.empty() && a[0] == '-') { fprintf(stderr, "Unknown option %s\n", a.c_str()); return 1; }
        else crit.prefixes.push_back(a);
    }

    // The kernel is compiled with __launch_bounds__(VANITY_MAX_TPB); a larger
    // block cannot launch at all, so say so here instead of letting it fail
    // later with a bare "too many resources requested for launch".
    if (tpb > VANITY_MAX_TPB) {
        fprintf(stderr, "Block size %d exceeds the kernel's limit, using %d.\n",
                tpb, VANITY_MAX_TPB);
        tpb = VANITY_MAX_TPB;
    }
    if (tpb < 32) { fprintf(stderr, "Block size must be at least 32.\n"); return 1; }

    cuda_check(cudaSetDevice(device), "setDevice");
    // Block (sleep) the host thread while waiting on the GPU instead of the
    // default busy-wait spin, which otherwise pegs one CPU core at 100% and
    // eats into the shared laptop power/thermal budget (lowering GPU boost).
    cuda_check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync), "setDeviceFlags");

    if (selftest) return run_selftest();
    if (benchmark) {
        if (grid_given || window)
            fprintf(stderr, "Note: --benchmark sweeps --window, --tpb and --blocks itself.\n");
        return run_benchmark();
    }

    if (crit.prefixes.empty() && crit.rules.empty()) {
        fprintf(stderr, "Give a hex prefix or a --repeat-* rule. Use --help.\n");
        return 1;
    }
    for (const std::string &pfx : crit.prefixes) {
        if (pfx.empty() || pfx.size() > 64) {
            fprintf(stderr, "Prefix must be 1-64 hex characters: '%s'\n", pfx.c_str());
            return 1;
        }
        for (char c : pfx)
            if (hexval(c) < 0) {
                fprintf(stderr, "Invalid hex character in prefix '%s': '%c'\n", pfx.c_str(), c);
                return 1;
            }
    }
    normalize(crit, true);
    const int npfx = (int)crit.prefixes.size();
    const int nrule = (int)crit.rules.size();

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

    const bool auto_blocks = blocks == 0;
    if (auto_blocks) blocks = wave_blocks(window, tpb, DEFAULT_WAVES);

    const double p_hit = hit_probability(crit);    // per-candidate chance of any match
    const unsigned long long threads = (unsigned long long)blocks * tpb;
    unsigned long long per_launch = threads * (unsigned long long)window;

    fprintf(stderr, "Searching for keys that match:\n");
    for (const std::string &p : crit.prefixes) {
        std::vector<uint8_t> req, mask;
        int len = build_matcher(p, req, mask);
        fprintf(stderr, "  prefix %-16s req %s mask %s\n", upper(p).c_str(),
                hex_upper(req.data(), len).c_str(), hex_upper(mask.data(), len).c_str());
    }
    for (RepeatRule r : crit.rules) {
        if (r.u == 1) fprintf(stderr, "  %d+ identical hex digits (%s, %s, ...)\n", r.n,
                              std::string(r.n, '0').c_str(), std::string(r.n, 'A').c_str());
        else {
            std::string ex; for (int k = 0; k < r.n / 2; k++) ex += "AB";
            fprintf(stderr, "  %d+ identical bytes (%s, ...)\n", r.n / 2, ex.c_str());
        }
    }
    fprintf(stderr, "Estimated attempts: 2^%.1f\n", -log2(p_hit));
    fprintf(stderr, "Grid: %d blocks%s x %d threads, window %d => %llu candidates/launch\n",
            blocks, auto_blocks ? " (" STR(DEFAULT_WAVES) " whole waves)" : "", tpb, window, per_launch);

    // Device buffers.
    uint8_t *d_scalar, *d_pub;
    unsigned long long *d_count, *d_units;
    cuda_check(cudaMalloc(&d_count, sizeof(unsigned long long)), "malloc count");
    cuda_check(cudaMalloc(&d_units, sizeof(unsigned long long) * RESULT_CAP), "malloc units");
    cuda_check(cudaMalloc(&d_scalar, 32), "malloc scalar");
    cuda_check(cudaMalloc(&d_pub, 32), "malloc pub");

    // Shared precomputed step table, sized for the starting window, so a
    // fallback to a smaller one can rebuild it in place.
    StepTable d_tab;
    if (!alloc_table(d_tab, window)) cuda_check(cudaGetLastError(), "malloc step table");

    // The filter for this set of criteria. Each combination is its own kernel,
    // so a mode never pays for checks it does not use.
    DevList dlist(npfx > 1 || (nrule && npfx) ? crit.prefixes : std::vector<std::string>());
    FilterOne f_one{0, 0};
    if (npfx) pack_prefix(crit.prefixes[0], f_one.req8, f_one.mask8);
    const FilterList f_list = dlist.filter();
    FilterRepeat<1> f_r1 = nrule == 1 ? make_repeat<1>(crit.rules) : FilterRepeat<1>{};
    FilterRepeat<2> f_r2 = nrule == 2 ? make_repeat<2>(crit.rules) : FilterRepeat<2>{};

    // Per-thread state: the random base scalar, the launch it was drawn at,
    // the carried centre point, and the "seed from base" flag.
    std::vector<uint8_t> h_bases(threads * 32);
    std::vector<unsigned long long> h_seeded(threads, 0);
    uint8_t *d_bases, *d_fresh;
    WalkState d_st;
    bignum25519 *d_dstep;
    cuda_check(cudaMalloc(&d_bases, threads * 32), "malloc bases");
    cuda_check(cudaMalloc(&d_fresh, threads), "malloc fresh");
    if (!alloc_state(d_st, threads)) cuda_check(cudaGetLastError(), "malloc walk state");
    cuda_check(cudaMalloc(&d_dstep, sizeof(bignum25519) * 3), "malloc dstep");

    unsigned long long launch_no = 0;
    // (Re)build everything that depends on the window, and give every thread a
    // fresh random base.
    auto setup_window = [&]() {
        build_step_table_kernel<<<1, 1>>>(d_tab, window);
        cuda_check(cudaGetLastError(), "build_step_table launch");
        build_launch_step_kernel<<<1, 1>>>(d_dstep, (unsigned long long)window);
        cuda_check(cudaGetLastError(), "build_launch_step launch");
        random_bases(h_bases.data(), threads);
        std::fill(h_seeded.begin(), h_seeded.end(), launch_no);
        cuda_check(cudaMemcpy(d_bases, h_bases.data(), threads * 32, cudaMemcpyHostToDevice),
                   "memcpy bases");
        cuda_check(cudaMemset(d_fresh, 1, threads), "memset fresh");
        cuda_check(cudaDeviceSynchronize(), "setup sync");
    };
    setup_window();

    const LaunchArgs la{d_bases, d_tab, d_count, d_units, d_dstep, d_st, d_fresh};
    auto launch = [&]() -> cudaError_t {
        if (nrule == 0)
            return npfx == 1 ? launch_vanity(window, blocks, tpb, la, f_one)
                             : launch_vanity(window, blocks, tpb, la, f_list);
        if (nrule == 1)
            return npfx == 0 ? launch_vanity(window, blocks, tpb, la, f_r1)
                             : launch_vanity(window, blocks, tpb, la,
                                             FilterEither<FilterRepeat<1>, FilterList>{f_r1, f_list});
        return npfx == 0 ? launch_vanity(window, blocks, tpb, la, f_r2)
                         : launch_vanity(window, blocks, tpb, la,
                                         FilterEither<FilterRepeat<2>, FilterList>{f_r2, f_list});
    };

    long found = 0;
    unsigned long long attempts = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto tlast = t0;

    while (true) {
        cuda_check(cudaMemset(d_count, 0, sizeof(unsigned long long)), "memset count");

        cudaError_t le = launch();
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
                // The step per launch changed, so the carried centres and the
                // step point are stale: rebuild and re-seed.
                setup_window();
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
        if (count > RESULT_CAP)
            fprintf(stderr, "\nWarning: %llu matches in one launch, only %d recorded. "
                            "Use a stricter criterion or fewer --blocks.\n",
                    count, RESULT_CAP);
        unsigned long long n_units = count < RESULT_CAP ? count : RESULT_CAP;
        std::vector<unsigned long long> units(n_units);
        if (n_units)
            cuda_check(cudaMemcpy(units.data(), d_units,
                                  sizeof(unsigned long long) * n_units, cudaMemcpyDeviceToHost),
                       "copy units");
        std::sort(units.begin(), units.end());       // groups hits by thread

        // At most one key per thread per launch, and the thread gets a new
        // random base right after: any two keys it produced from one base
        // would differ by a known small multiple of 8, so leaking one private
        // key would give away the other. Keys from different bases are
        // independent. The discarded hits only matter for criteria so loose
        // that one thread meets them several times in a single window.
        std::vector<unsigned long long> reseed;
        for (unsigned long long u : units) {
            const unsigned long long g = u / window, j = u % window;
            if (!reseed.empty() && reseed.back() == g) continue;

            uint8_t scalar[32];
            scalar_add_u64(scalar, &h_bases[g * 32],
                           ((launch_no - h_seeded[g]) * window + j) * 8ULL);
            clamp_scalar(scalar);

            uint8_t pub[32], nib[64];
            pack_one(scalar, pub, d_scalar, d_pub);
            key_nibbles(pub, nib);

            // Host re-check against the full criteria (defensive; also
            // validates the GPU, and tells us which criterion actually matched).
            int which = match_nibbles(crit, nib, 64);
            if (which < 0) continue;
            reseed.push_back(g);

            uint8_t signing[32];
            fill_random(signing, 32);
            if (progress) fputc('\r', stderr);
            printf("\nFound matching key!\n");
            if (which < npfx) {
                if (npfx + nrule > 1) printf("Prefix:      %s\n", upper(crit.prefixes[which]).c_str());
            } else {
                RepeatRule r = crit.rules[which - npfx];
                int run = periodic_len(nib, 64, r.u);
                if (r.u == 1) printf("Repeat:      %d identical hex digits\n", run);
                else          printf("Repeat:      %d identical bytes\n", run / 2);
            }
            printf("Public Key:  %s\n", hex_upper(pub, 32).c_str());
            printf("Private Key: %s%s\n", hex_upper(scalar, 32).c_str(),
                   hex_upper(signing, 32).c_str());
            fflush(stdout);

            if (++found >= limit && limit != 0) goto done;
        }

        attempts += per_launch;
        launch_no++;
        if (!reseed.empty()) {
            for (unsigned long long g : reseed) {
                random_bases(&h_bases[g * 32], 1);
                h_seeded[g] = launch_no;
            }
            if (reseed.size() <= 64) {
                const uint8_t one = 1;
                for (unsigned long long g : reseed) {
                    cuda_check(cudaMemcpy(d_bases + g * 32, &h_bases[g * 32], 32,
                                          cudaMemcpyHostToDevice), "memcpy base");
                    cuda_check(cudaMemcpy(d_fresh + g, &one, 1, cudaMemcpyHostToDevice),
                               "memcpy fresh");
                }
            } else {
                // Every other thread's flag is already 0: the kernel clears it.
                std::vector<uint8_t> fl(threads, 0);
                for (unsigned long long g : reseed) fl[g] = 1;
                cuda_check(cudaMemcpy(d_bases, h_bases.data(), threads * 32,
                                      cudaMemcpyHostToDevice), "memcpy bases");
                cuda_check(cudaMemcpy(d_fresh, fl.data(), threads, cudaMemcpyHostToDevice),
                           "memcpy fresh");
            }
        }

        if (progress) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - tlast).count() >= 0.3) {
                double secs = std::chrono::duration<double>(now - t0).count();
                double mps = secs > 0 ? attempts / secs / 1e6 : 0;
                char line[128];
                int n = snprintf(line, sizeof line, "Tried %llu keys (%.1f Mkeys/s)",
                                 attempts, mps);
                if (found == 0 && n > 0 && n < (int)sizeof line) {
                    // Chance the span already searched contained a match. Each
                    // candidate is a distinct scalar meeting some criterion
                    // with probability p_hit, so P = 1 - (1-p_hit)^attempts.
                    // expm1 and log1p keep that accurate when p_hit is tiny and
                    // the product would underflow a plain pow.
                    double hit = -expm1((double)attempts * log1p(-p_hit));
                    snprintf(line + n, sizeof line - n,
                             ", %.3g%% chance it was already in range", hit * 100.0);
                }
                // Left-pad to a fixed width: the line shrinks once a key is
                // found, and a bare \r would leave the old tail on screen.
                fprintf(stderr, "\r%-76s", line);
                tlast = now;
            }
        }
    }
done:
    fprintf(stderr, "\n");
    return 0;
}

// =========================================================================
// Self-tests
// =========================================================================

// Each vanity_kernel<W> instantiation pins its own local-memory reserve for the
// lifetime of the context, so running several in one context piles them up and
// can exhaust a small or busy GPU. Give every kernel test a fresh context, the
// same way bench_one does, and report a genuine shortage as a skip rather than
// failing a correctness test for an environmental reason.
static void fresh_context() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceReset();
    cudaSetDevice(dev);
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
}

// Everything one search-kernel configuration needs, for the tests and the
// benchmark. Allocation failures clear `ok` instead of exiting.
struct Rig {
    int window, blocks, tpb;
    unsigned long long thr;
    uint8_t *d_bases = nullptr, *d_fresh = nullptr;
    unsigned long long *d_count = nullptr, *d_units = nullptr;
    StepTable tab{nullptr, nullptr, nullptr, nullptr};
    WalkState st{nullptr, nullptr, nullptr};
    bignum25519 *d_dstep = nullptr;
    std::vector<uint8_t> bases;          // per-thread base scalars (host copy)
    bool ok = true;                      // allocations and precompute succeeded
    bool oom = false;                    // a launch ran out of memory
    unsigned long long last_count = 0;   // raw hit count of the last run

    Rig(int w, int b, int t) : window(w), blocks(b), tpb(t), thr((unsigned long long)b * t) {
        auto m = [&](void *p, size_t n) {
            if (ok && cudaMalloc((void **)p, n) != cudaSuccess) { cudaGetLastError(); ok = false; }
        };
        m(&d_bases, thr * 32); m(&d_fresh, thr);
        m(&d_count, sizeof(unsigned long long));
        m(&d_units, sizeof(unsigned long long) * RESULT_CAP);
        if (ok && !alloc_table(tab, w)) { cudaGetLastError(); ok = false; }
        if (ok && !alloc_state(st, thr)) { cudaGetLastError(); ok = false; }
        m(&d_dstep, sizeof(bignum25519) * 3);
        bases.resize(thr * 32);
        random_bases(bases.data(), thr);
        if (!ok) return;
        build_step_table_kernel<<<1, 1>>>(tab, w);
        build_launch_step_kernel<<<1, 1>>>(d_dstep, (unsigned long long)w);
        if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); ok = false; return; }
        upload();
    }
    ~Rig() {
        cudaFree(d_bases); cudaFree(d_fresh); cudaFree(d_count); cudaFree(d_units);
        free_table(tab); free_state(st); cudaFree(d_dstep);
    }
    Rig(const Rig &) = delete;
    Rig &operator=(const Rig &) = delete;

    // Copy `bases` to the device.
    void upload() {
        cudaMemcpy(d_bases, bases.data(), thr * 32, cudaMemcpyHostToDevice);
    }

    template <class F>
    cudaError_t launch(const F &f) {
        cudaMemset(d_count, 0, sizeof(unsigned long long));
        LaunchArgs la{d_bases, tab, d_count, d_units, d_dstep, st, d_fresh};
        return launch_vanity(window, blocks, tpb, la, f);
    }

    // One launch; returns the sorted recorded units. reseed = start every
    // thread from its base (otherwise continue from the carried centres).
    template <class F>
    std::vector<unsigned long long> run(const F &f, bool reseed = true) {
        std::vector<unsigned long long> v;
        if (!ok || oom) return v;
        if (reseed) cudaMemset(d_fresh, 1, thr);
        cudaError_t le = launch(f);
        if (le == cudaErrorMemoryAllocation) { cudaGetLastError(); oom = true; return v; }
        cuda_check(le, "test launch");
        cuda_check(cudaDeviceSynchronize(), "test sync");
        cuda_check(cudaMemcpy(&last_count, d_count, sizeof(last_count), cudaMemcpyDeviceToHost),
                   "test count");
        v.resize(last_count < RESULT_CAP ? last_count : RESULT_CAP);
        if (!v.empty())
            cuda_check(cudaMemcpy(v.data(), d_units, sizeof(unsigned long long) * v.size(),
                                  cudaMemcpyDeviceToHost), "test units");
        std::sort(v.begin(), v.end());
        return v;
    }

    bool skipped(const char *what) const {
        if (ok && !oom) return false;
        printf("[selftest] window %5d %s: skipped (not enough free VRAM)\n", window, what);
        return true;
    }
};

// -------------------------------------------------------------------------
// Self-test: key_lo64 on multiply outputs must give the low 8 bytes of the
// canonical encoding, for products of random field elements and of the
// canonicalisation edge cases (0, 1, p-1, p, p+1, 2^255-1). The one allowed
// disagreement is the documented miss: canonical y below 2^52 + 19, where the
// multiply can leave y + p. Real candidates hit that with probability ~2^-203;
// here the inputs = 0 mod p and the other edge cases produce it on purpose.
// -------------------------------------------------------------------------
static int check_partial_products() {
    unsigned long long *d_cnt, h[5];
    cuda_check(cudaMalloc(&d_cnt, sizeof h), "malloc partial cnt");
    cuda_check(cudaMemset(d_cnt, 0, sizeof h), "memset partial cnt");
    unsigned long long seed = 0;
    fill_random((uint8_t *)&seed, sizeof seed);
    partial_mul_kernel<<<256, 256>>>(d_cnt, 256, seed);
    cuda_check(cudaGetLastError(), "partial launch");
    cuda_check(cudaDeviceSynchronize(), "partial sync");
    cuda_check(cudaMemcpy(h, d_cnt, sizeof h, cudaMemcpyDeviceToHost), "copy partial cnt");
    cudaFree(d_cnt);
    const unsigned long long bad = h[1] + h[2];
    printf("[selftest] partial final products == full multiply on %llu kernel-shaped products: %s"
           " (%llu ambiguous 26-bit cases, %llu needing Q+1)\n",
           h[0], bad ? "BROKEN" : "exact", h[3], h[4]);
    return bad ? 1 : 0;
}

static int check_key_lo64() {
    const uint32_t m26 = (1u << 26) - 1, m25 = (1u << 25) - 1;
    typedef std::array<uint32_t, 10> Fe;
    auto full = [&](uint32_t lo0) {   // limb0 = lo0, all higher limbs saturated
        Fe f{};
        f[0] = lo0;
        for (int k = 1; k < 10; k++) f[k] = (k & 1) ? m25 : m26;
        return f;
    };
    // p = 2^255 - 19 is limb0 = 2^26 - 19 = m26 - 18 with every other limb full.
    std::vector<Fe> edge = {Fe{}, Fe{1}, full(m26 - 19) /* p-1 */, full(m26 - 18) /* p */,
                            full(m26 - 17) /* p+1 */, full(m26) /* 2^255-1 */};
    std::vector<Fe> va, vb;
    for (const Fe &x : edge) for (const Fe &y : edge) { va.push_back(x); vb.push_back(y); }
    std::mt19937 rng(12345);
    auto rnd = [&]() { Fe f; for (int k = 0; k < 10; k++) f[k] = rng() & ((k & 1) ? m25 : m26); return f; };
    for (int i = 0; i < 4096; i++) { va.push_back(rnd()); vb.push_back(i & 1 ? rnd() : edge[(i / 2) % 6]); }

    const int n = (int)va.size();
    uint32_t *d_a, *d_b; uint8_t *d_out32; unsigned long long *d_lo;
    cuda_check(cudaMalloc(&d_a, sizeof(Fe) * n), "malloc lo64 a");
    cuda_check(cudaMalloc(&d_b, sizeof(Fe) * n), "malloc lo64 b");
    cuda_check(cudaMalloc(&d_out32, 32 * n), "malloc lo64 out32");
    cuda_check(cudaMalloc(&d_lo, sizeof(unsigned long long) * n), "malloc lo64 lo");
    cuda_check(cudaMemcpy(d_a, va.data(), sizeof(Fe) * n, cudaMemcpyHostToDevice), "memcpy lo64 a");
    cuda_check(cudaMemcpy(d_b, vb.data(), sizeof(Fe) * n, cudaMemcpyHostToDevice), "memcpy lo64 b");
    mul_lo64_kernel<<<(n + 127) / 128, 128>>>((const bignum25519 *)d_a, (const bignum25519 *)d_b,
                                              n, d_out32, d_lo);
    cuda_check(cudaGetLastError(), "lo64 launch");
    cuda_check(cudaDeviceSynchronize(), "lo64 sync");

    std::vector<uint8_t> out32(32 * n);
    std::vector<unsigned long long> lo(n);
    cuda_check(cudaMemcpy(out32.data(), d_out32, 32 * n, cudaMemcpyDeviceToHost), "copy out32");
    cuda_check(cudaMemcpy(lo.data(), d_lo, sizeof(unsigned long long) * n, cudaMemcpyDeviceToHost), "copy lo");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out32); cudaFree(d_lo);

    int fails = 0, small = 0;
    for (int i = 0; i < n; i++) {
        const uint8_t *c = &out32[i * 32];
        unsigned long long want = 0;
        for (int b = 0; b < 8; b++) want |= (unsigned long long)c[b] << (8 * b);
        if (want == lo[i]) continue;
        bool tiny = want < (1ULL << 52) + 19;
        for (int b = 8; b < 32; b++) tiny = tiny && c[b] == 0;
        if (tiny) small++; else fails++;
    }
    printf("[selftest] filter bits (full and partial multiply) == low 8 bytes of contract: %d/%d products ok"
           " (+%d forced y < 2^52 left as y + p, the documented miss)\n",
           n - fails - small, n, small);
    return fails;
}

// -------------------------------------------------------------------------
// Self-test: window coverage. Run the real search kernel with an empty filter,
// so every candidate "matches" and out_units becomes the exact set of units the
// window walked. It must be precisely [0, blocks*tpb*W) — no gap, no repeat, no
// overrun into the neighbouring thread's span. This is what guards the +/-i
// walk, whose failure mode is silently losing or duplicating candidates rather
// than producing wrong keys.
// -------------------------------------------------------------------------
static int check_window_coverage(int window, int blocks, int tpb) {
    const unsigned long long expect = (unsigned long long)blocks * tpb * window;
    if (expect > RESULT_CAP) { printf("[selftest] coverage W=%d skipped (too many units)\n", window); return 0; }
    fresh_context();
    Rig r(window, blocks, tpb);
    std::vector<unsigned long long> units = r.run(FilterOne{0, 0});
    if (r.skipped("coverage")) return 0;

    bool ok = (r.last_count == expect) && (units.size() == expect);
    for (unsigned long long i = 0; ok && i < expect; i++) if (units[i] != i) ok = false;
    printf("[selftest] window %5d coverage (%d x %d threads): %s (%llu/%llu units)\n",
           window, blocks, tpb, ok ? "exact" : "BROKEN", r.last_count, expect);
    return ok ? 0 : 1;
}

// -------------------------------------------------------------------------
// Self-test: the persistent walk. Between launches a thread does not rebuild
// its window centre from its base; it advances the point it kept by D = W*8*B.
// If that drifts, the search silently scans the wrong scalars — nothing else
// would catch it, because the recorded units stay in range either way.
//
// So: run the kernel with every base moved on by one window from a cold seed,
// and separately run it at the original bases and then once more off the
// carried-over state. With a real (1-byte) prefix the recorded set depends on
// the actual points, and the two must agree exactly.
// -------------------------------------------------------------------------
static int check_persistent_walk(int window, int blocks, int tpb) {
    fresh_context();
    Rig r(window, blocks, tpb);
    const FilterOne f{0x00, 0xFF};                 // one byte: ~1 hit in 256 candidates

    std::vector<uint8_t> base1 = r.bases, base2(base1.size());
    for (unsigned long long g = 0; g < r.thr; g++) {
        scalar_add_u64(&base2[g * 32], &base1[g * 32], (unsigned long long)window * 8ULL);
        clamp_scalar(&base2[g * 32]);
    }

    r.bases = base2; r.upload();
    std::vector<unsigned long long> cold = r.run(f);           // seeded straight at base2
    r.bases = base1; r.upload();
    (void)r.run(f);                                            // seed at base1, carry state
    std::vector<unsigned long long> warm = r.run(f, false);    // must land on base2 by walking
    if (r.skipped("persistent walk")) return 0;

    bool ok = !cold.empty() && cold == warm;
    printf("[selftest] window %5d persistent walk == cold seed: %s (%zu vs %zu hits)\n",
           window, ok ? "identical" : "BROKEN", cold.size(), warm.size());
    return ok ? 0 : 1;
}

// -------------------------------------------------------------------------
// Self-test: ground truth. The other kernel tests compare the kernel with
// itself (cold seed vs carried walk, list vs single prefix), so an addition
// formula that produced consistent but wrong y values would pass all of them.
// Here every candidate of a small grid is recomputed independently — full
// fixed-base scalar multiplication, projective, packed by donna — and the set
// of candidates whose key has a given hex digit must be exactly the set the
// kernel records: nothing missed, nothing invented. Checked for a cold launch
// and for the next one off the carried state.
// -------------------------------------------------------------------------
static int check_ground_truth(int window, int blocks, int tpb) {
    fresh_context();
    Rig r(window, blocks, tpb);
    const FilterOne f{0x0A, 0x0F};                  // second hex digit is A: 1 in 16
    uint8_t *d_scalar, *d_pub;
    cuda_check(cudaMalloc(&d_scalar, 32), "gt malloc scalar");
    cuda_check(cudaMalloc(&d_pub, 32), "gt malloc pub");

    int fails = 0;
    size_t hits = 0;
    for (int launch = 0; launch < 2; launch++) {
        std::vector<unsigned long long> got = r.run(f, launch == 0);
        if (r.skipped("ground truth")) { cudaFree(d_scalar); cudaFree(d_pub); return 0; }
        std::vector<unsigned long long> want;
        for (unsigned long long g = 0; g < r.thr; g++)
            for (int j = 0; j < window; j++) {
                uint8_t scalar[32], pub[32];
                scalar_add_u64(scalar, &r.bases[g * 32],
                               ((unsigned long long)launch * window + j) * 8ULL);
                clamp_scalar(scalar);
                pack_one(scalar, pub, d_scalar, d_pub);
                if ((pub[0] & 0x0F) == 0x0A) want.push_back(g * window + j);
            }
        hits += want.size();
        if (want.empty() || got != want) fails++;
    }
    cudaFree(d_scalar); cudaFree(d_pub);
    printf("[selftest] window %5d kernel == independent s*B for %llu candidates x 2 launches: %s (%zu hits)\n",
           window, r.thr * (unsigned long long)window, fails ? "BROKEN" : "exact", hits);
    return fails;
}

// -------------------------------------------------------------------------
// Self-test: a filter against a plain prefix list that means the same thing.
// Both kernels run over the same bases and grid, so the recorded sets must be
// identical: nothing missed, nothing invented.
// -------------------------------------------------------------------------
// `make_filter(extra)` builds the filter under test; `extra` is `extra_prefixes`
// already in device memory, created after the context reset like everything
// else here.
template <class MakeFilter>
static int check_filter_vs_list(const char *name, int window, int blocks, int tpb,
                                const std::vector<std::string> &extra_prefixes,
                                MakeFilter make_filter, const std::vector<std::string> &equiv) {
    fresh_context();
    Rig r(window, blocks, tpb);
    std::vector<unsigned long long> got, want;
    {
        DevList extra(extra_prefixes), dl(equiv);
        got = r.run(make_filter(extra));
        want = r.run(dl.filter());
    }
    if (r.skipped(name)) return 0;
    bool ok = !want.empty() && got == want;
    printf("[selftest] window %5d %s: %s (%zu vs %zu hits)\n",
           window, name, ok ? "identical" : "BROKEN", got.size(), want.size());
    return ok ? 0 : 1;
}

// -------------------------------------------------------------------------
// Self-test: the prefix-list kernel. Searching a list must find exactly the
// union of what the single-prefix kernel finds for each entry separately.
// -------------------------------------------------------------------------
static int check_prefix_list(int window, int blocks, int tpb) {
    fresh_context();
    Rig r(window, blocks, tpb);
    // Deliberately mixed lengths (2, 1 and 4 digits) so the masks differ too —
    // with uniform masks a filter that ignored the per-entry mask would still
    // pass. The short one dominates the hit rate and keeps the sample healthy.
    const std::vector<std::string> entries = {"37", "c", "abcd"};

    std::vector<unsigned long long> list, uni;
    {
        DevList dl(entries);
        list = r.run(dl.filter());
    }
    for (const std::string &e : entries) {
        FilterOne f; pack_prefix(e, f.req8, f.mask8);
        std::vector<unsigned long long> one = r.run(f);
        uni.insert(uni.end(), one.begin(), one.end());
    }
    std::sort(uni.begin(), uni.end());
    uni.erase(std::unique(uni.begin(), uni.end()), uni.end());
    if (r.skipped("prefix list")) return 0;

    bool ok = !uni.empty() && list == uni;
    printf("[selftest] window %5d list of %zu == union of single runs: %s (%zu vs %zu hits)\n",
           window, entries.size(), ok ? "identical" : "BROKEN", list.size(), uni.size());
    return ok ? 0 : 1;
}

// Every key a repeat rule accepts, as explicit prefixes.
static std::vector<std::string> expand_rule(RepeatRule r) {
    static const char *H = "0123456789abcdef";
    std::vector<std::string> out;
    const int units = r.u == 1 ? 16 : 256;
    for (int v = 0; v < units; v++) {
        std::string unit = r.u == 1 ? std::string(1, H[v]) : std::string{H[v >> 4], H[v & 15]};
        std::string s;
        while ((int)s.size() < r.n) s += unit;
        out.push_back(s.substr(0, r.n));
    }
    return out;
}

// -------------------------------------------------------------------------
// Self-test: the repeat filters, alone and combined with a prefix list,
// against the equivalent explicit list (itself checked by check_prefix_list).
// The nibble rule uses an odd length on purpose: its last digit is a high
// nibble, the case where the digits are not contiguous in the low word.
// -------------------------------------------------------------------------
static int check_repeat(int window, int blocks, int tpb) {
    int fails = 0;
    const std::vector<RepeatRule> nib = {{3, 1}}, both = {{3, 1}, {4, 2}};
    const std::vector<std::string> extra = {"37", "c0d"};

    fails += check_filter_vs_list("repeat-nibble 3 == its 16 prefixes", window, blocks, tpb, {},
                                  [&](const DevList &) { return make_repeat<1>(nib); },
                                  expand_rule(nib[0]));

    std::vector<std::string> equiv = expand_rule(both[0]);
    for (const std::string &s : expand_rule(both[1])) equiv.push_back(s);
    for (const std::string &s : extra) equiv.push_back(s);
    fails += check_filter_vs_list("repeats + list == their 274 prefixes", window, blocks, tpb, extra,
                                  [&](const DevList &dl) {
                                      return FilterEither<FilterRepeat<2>, FilterList>{
                                          make_repeat<2>(both), dl.filter()};
                                  },
                                  equiv);
    return fails;
}

// -------------------------------------------------------------------------
// Self-test: normalize() and hit_probability() against brute force. Every key
// of 5 hex digits is enumerated and checked against the raw criteria; the
// matching fraction must equal the computed probability exactly, which checks
// the overlap arithmetic and that dropping redundant criteria changes nothing.
// The host-side matcher used for real hits is the one enumerated here.
// -------------------------------------------------------------------------
static int check_probability() {
    struct Case { std::vector<std::string> p; std::vector<RepeatRule> r; };
    const Case cases[] = {
        {{"a", "ab", "37", "37"}, {}},
        {{"aa", "aab", "5", "12", "666"}, {{3, 1}}},
        {{"a", "0a0", "77", "2", "5555"}, {{3, 1}, {4, 2}}},
        {{"3434", "343", "1"}, {{5, 1}, {4, 2}}},
        {{"ab", "abab"}, {{4, 2}}},
    };
    const int D = 5, N = 1 << (4 * D);
    int fails = 0, idx = 0;
    for (const Case &cs : cases) {
        Criteria raw{cs.p, cs.r}, norm = raw;
        normalize(norm, false);
        long hits = 0, hits_norm = 0;
        uint8_t nib[D];
        for (int x = 0; x < N; x++) {
            for (int i = 0; i < D; i++) nib[i] = (x >> (4 * (D - 1 - i))) & 0xF;
            hits += match_nibbles(raw, nib, D) >= 0;
            hits_norm += match_nibbles(norm, nib, D) >= 0;
        }
        const double exact = (double)hits / N, calc = hit_probability(norm);
        const bool ok = hits == hits_norm && fabs(exact - calc) <= 1e-12 * exact;
        if (!ok) {
            printf("[selftest] probability case %d: BROKEN (brute %ld/%d, normalized %ld, "
                   "formula %.10g)\n", idx, hits, N, hits_norm, calc * N);
            fails++;
        }
        idx++;
    }
    printf("[selftest] hit probability == brute force over 16^%d keys: %s (%d cases)\n",
           D, fails ? "BROKEN" : "exact", idx);
    return fails;
}

// -------------------------------------------------------------------------
// Self-tests: incremental identity on device + a fixed known-answer vector.
// -------------------------------------------------------------------------
static int run_selftest() {
    const int N = 512;
    std::vector<uint8_t> scalars(N * 32);
    random_bases(scalars.data(), N);

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

    // crypto_scalarmult_ed25519_base_noclamp(kat_scalar) from libsodium/PyNaCl.
    const std::string kat_want =
        "CFE058A4A189EE7230E43A1347EA1A7EEF01F3557991A7FD3CEC8915FD290AEC";
    const std::string kat_got = hex_upper(pub0, 32);
    const bool kat_ok = kat_got == kat_want;
    printf("[selftest] KAT scalar*B == libsodium: %s\n", kat_ok ? "ok" : "BROKEN");
    if (!kat_ok) printf("[selftest]   got  %s\n[selftest]   want %s\n",
                        kat_got.c_str(), kat_want.c_str());
    fails += !kat_ok;

    fails += check_key_lo64();
    fails += check_partial_products();
    fails += check_probability();
    // A power-of-two window, a "half" window, and >1 thread so the boundary
    // between neighbouring spans is actually exercised.
    fails += check_window_coverage(64, 2, 2);
    fails += check_window_coverage(1024, 1, 2);
    fails += check_window_coverage(1536, 1, 2);
    fails += check_ground_truth(256, 1, 4);
    fails += check_ground_truth(1536, 1, 2);
    fails += check_persistent_walk(256, 8, 64);
    fails += check_persistent_walk(1536, 4, 32);
    fails += check_prefix_list(256, 8, 64);
    fails += check_prefix_list(1536, 4, 32);
    fails += check_repeat(256, 8, 64);
    fails += check_repeat(1536, 4, 32);

    printf("[selftest] %s\n", fails ? "FAILED" : "all passed");
    return fails == 0 ? 0 : 2;
}

// -------------------------------------------------------------------------
// One benchmark point: fresh context, build the step table, ~1s warm-up + ~1.5s
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

    Rig r(w, blocks, tpb);
    if (!r.ok) return -1;
    cudaMemset(r.d_fresh, 1, r.thr);
    const FilterOne f{0, ~0ULL};             // 8 bytes of zeros: never matches
    const unsigned long long per_launch = r.thr * (unsigned long long)w;

    // Launches until >= target seconds elapse; returns elapsed (or -1 on error)
    // with the candidate count via out-param. Only the very first launch pays
    // for the seed: the kernel clears each thread's flag.
    auto run_span = [&](double target, unsigned long long &cand) -> double {
        cand = 0;
        auto t0 = clock::now();
        double el = 0;
        do {
            if (r.launch(f) != cudaSuccess) { cudaGetLastError(); return -1; }
            if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); return -1; }
            cand += per_launch;
            el = std::chrono::duration<double>(clock::now() - t0).count();
        } while (el < target);
        return el;
    };

    unsigned long long tmp = 0;
    if (run_span(1.0, tmp) < 0) return -1;         // warm-up doubles as a fit check
    unsigned long long cand = 0;
    double el = run_span(1.5, cand);
    return (el > 0) ? (double)cand / el / 1e6 : 0;
}

// -------------------------------------------------------------------------
// Benchmark: find the fastest (window, tpb, blocks) for this GPU in three
// sweeps, each around the best of the previous one (~1 minute in total):
//   1. every window, at the default tpb and grid;
//   2. tpb over the two fastest windows (two, because the ranking of windows is
//      not separable from tpb);
//   3. the number of whole waves for the best pair.
// Grids are always whole waves (see wave_blocks); an explicit --blocks that is
// not a multiple of SM count x blocks per SM wastes part of the last wave.
// -------------------------------------------------------------------------
static int run_benchmark() {
    int numSM = 1, maxThreadsSM = 1, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&maxThreadsSM, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    fprintf(stderr, "Benchmarking %s (%d SM), ~1 minute...\n", prop.name, numSM);

    struct Pt { int w, tpb, waves, blocks; double mps; };
    Pt best{0, 0, 0, 0, -1};
    auto measure = [&](int w, int t, int waves) {
        Pt p{w, t, waves, wave_blocks(w, t, waves), 0};
        p.mps = bench_one(w, p.blocks, t, dev);
        if (p.mps > best.mps) best = p;
        return p;
    };

    // 1. Windows.
    printf("%6s  %10s  %8s  %9s   (tpb %d, %d waves)\n", "window", "Mkeys/s", "loc/thr",
           "reserve", DEFAULT_TPB, DEFAULT_WAVES);
    double w1 = -1, w2 = -1; int win1 = 0, win2 = 0;
    const int nW = (int)(sizeof(kWindows) / sizeof(kWindows[0]));
    for (int idx = nW - 1; idx >= 0; idx--) {      // ascending, small windows first
        const int w = kWindows[idx];
        Pt p = measure(w, DEFAULT_TPB, DEFAULT_WAVES);
        size_t localB = window_local_bytes(w);     // context is alive after bench_one
        size_t reserveMB = window_reserve_bytes(w, numSM, maxThreadsSM) >> 20;
        if (p.mps < 0)
            printf("%6d  %10s  %6zuKB  %6zuMB\n", w, "OOM/skip", localB >> 10, reserveMB);
        else {
            printf("%6d  %10.1f  %6zuKB  %6zuMB\n", w, p.mps, localB >> 10, reserveMB);
            if (p.mps > w1) { w2 = w1; win2 = win1; w1 = p.mps; win1 = w; }
            else if (p.mps > w2) { w2 = p.mps; win2 = w; }
        }
        fflush(stdout);
    }
    if (!win1) { printf("No window fits this GPU.\n"); return 1; }

    // 2. Block size over the two fastest windows. VANITY_MAX_TPB is the ceiling
    // (see __launch_bounds__); within the range the best is not monotone.
    printf("\n%8s  %6s  %7s  %10s   (%d waves)\n", "window", "tpb", "blocks", "Mkeys/s", DEFAULT_WAVES);
    for (int w : {win1, win2}) {
        if (!w) continue;
        for (int t : {64, DEFAULT_TPB, 256, VANITY_MAX_TPB}) {
            Pt p = t == DEFAULT_TPB ? Pt{w, t, DEFAULT_WAVES, wave_blocks(w, t, DEFAULT_WAVES),
                                         w == win1 ? w1 : w2}
                                    : measure(w, t, DEFAULT_WAVES);
            if (p.mps < 0) printf("%8d  %6d  %7d  %10s\n", w, t, p.blocks, "skip");
            else printf("%8d  %6d  %7d  %10.1f\n", w, t, p.blocks, p.mps);
            fflush(stdout);
        }
    }

    // 3. Number of waves for the best pair.
    const Pt pair = best;
    printf("\n%8s  %6s  %6s  %7s  %10s\n", "window", "tpb", "waves", "blocks", "Mkeys/s");
    for (int waves : {8, 16, DEFAULT_WAVES, 64}) {
        Pt p = waves == pair.waves ? pair : measure(pair.w, pair.tpb, waves);
        if (p.mps < 0) printf("%8d  %6d  %6d  %7d  %10s\n", p.w, p.tpb, waves, p.blocks, "skip");
        else printf("%8d  %6d  %6d  %7d  %10.1f\n", p.w, p.tpb, waves, p.blocks, p.mps);
        fflush(stdout);
    }

    cudaDeviceReset();
    cudaSetDevice(dev);
    printf("\n(reserve = worst-case local memory the driver pins = "
           "SM count x maxThreadsPerSM x loc/thr)\n");
    printf("Fastest: --window %d --tpb %d --blocks %d  (%.1f Mkeys/s)\n",
           best.w, best.tpb, best.blocks, best.mps);
    printf("Differences under ~1%% are within run-to-run noise.\n");
    return 0;
}
