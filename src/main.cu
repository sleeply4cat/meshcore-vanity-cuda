// meshcore-vanity-cuda — GPU vanity public-key search for MeshCore.
//
// Algorithm (see README): pubkey(s+8) = pubkey(s) + 8*B, so instead of a full
// fixed-base scalar multiplication per candidate we do ONE point addition
// (P += 8B) and amortize the field inversion over a window of WINDOW
// candidates with Montgomery's batch-inversion trick.
//
// Each thread g owns candidates s = base + (g*WINDOW + j)*8, j in [0,WINDOW).
// One fixed-base multiply computes its window start P0 = (base + g*WINDOW*8)*B,
// then it walks P += 8B, storing (y,z) for every point, batch-inverts the z's,
// and prefix-checks each affine y.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <random>
#include <chrono>

#include "ed25519.cuh"

// Window = candidates per thread per launch (batch size). It is chosen at
// runtime (--window, or auto-fit to VRAM); the kernel is templated on it so the
// per-thread buffers stay compile-time sized. MAXW bounds the precomputed table.
#define MAXW 1024
static const int kWindows[] = {1024, 512, 256, 128, 64};  // supported, descending

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
// Precompute the step table: affine coords of i*D (D = 8B) for i in [1,WINDOW),
// plus P_i = x_i*y_i (so the hot loop needs no extra mul for the denominator).
// These points are identical for every thread, so this runs once and the main
// kernel just reads the table (broadcast across the warp). Single thread; the
// per-point inversion cost is one-time and negligible.
// -------------------------------------------------------------------------
__global__ void build_step_table_kernel(bignum25519 *gx, bignum25519 *gy, bignum25519 *gp, int window) {
    ge25519_niels D; load_step_8B(&D);
    uint8_t eight[32]; for (int k = 0; k < 32; k++) eight[k] = 0; eight[0] = 8;
    ge25519 Q; scalar_to_point(&Q, eight);   // Q = 8B = 1*D
    for (int i = 1; i < window; i++) {
        ge_to_affine(gx[i], gy[i], &Q);
        curve25519_mul(gp[i], gx[i], gy[i]);
        if (i + 1 < window) ge25519_nielsadd2(&Q, &D);   // Q += D
    }
}

// -------------------------------------------------------------------------
// Main search kernel — affine batched-addition walk (only the y-coordinate).
//
// For candidate i (scalar s0 + i*8) the point is P0 + i*D. Using the complete
// twisted-Edwards (a=-1) addition and keeping only y:
//     y_i = (x0*x_i + y0*y_i) / (1 - d*x0*y0 * x_i*y_i)
// where (x0,y0) is the window-start point (one fixed-base multiply per thread)
// and (x_i, y_i, P_i=x_i*y_i) come from the shared precomputed step table.
// With K = d*x0*y0 (once per thread), each candidate costs ~7 field muls
// (2 for the numerator, 1 for the denominator, 3 amortized for the shared
// Montgomery inversion, 1 for y=num*inv) versus ~11 for the projective walk —
// and the x-coordinate is never computed.
//
// Register/occupancy tuning (maxrregcount, __launch_bounds__) was benchmarked
// and is neutral-to-worse: ALU-bound, high ILP saturates the int units even at
// low occupancy, so plain launch is best.
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
    const unsigned long long gid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long start_unit = gid * (unsigned long long)W;

    // Cache the small prefix (<=32 bytes).
    uint8_t lreq[32], lmask[32];
    for (int i = 0; i < prefix_len; i++) { lreq[i] = req[i]; lmask[i] = mask[i]; }

    // Window start scalar = base + start_unit*8, then its affine point.
    uint8_t s0[32];
    scalar_add_u64(s0, base, start_unit * 8ULL);
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
        unsigned char pub[32];
        curve25519_contract(pub, y);
        // Matches the low 255 bits of y (bit 255 = x parity is ignored; the
        // host recomputes the full compressed key for any hit).
        for (int i = 0; i < prefix_len; i++)
            if ((pub[i] & lmask[i]) != lreq[i]) return;
        unsigned long long slot = atomicAdd(out_count, 1ULL);
        if (slot < RESULT_CAP) out_units[slot] = unit;
    };

    // Candidate i=0 is P0 itself: y = y0.
    check_and_record(y0, start_unit);

    // Candidates i=1..W-1: numerator/denominator per point, then one
    // shared Montgomery inversion of all denominators.
    bignum25519 num[W], den[W], pref[W];
    bignum25519 acc; curve25519_copy(acc, one);
    for (int i = 1; i < W; i++) {
        bignum25519 a, b, c;
        curve25519_mul(a, x0, gx[i]);   // x0 * x_i
        curve25519_mul(b, y0, gy[i]);   // y0 * y_i
        curve25519_add_reduce(num[i], a, b);           // num = x0 x_i + y0 y_i
        curve25519_mul(c, K, gp[i]);    // c = d x0 y0 x_i y_i
        curve25519_sub_reduce(den[i], one, c);         // den = 1 - c
        curve25519_copy(pref[i], acc);                 // prefix product
        curve25519_mul(acc, acc, den[i]);
    }
    curve25519_recip(acc, acc);                        // 1 / prod(den)

    for (int i = W - 1; i >= 1; i--) {
        bignum25519 inv, y;
        curve25519_mul(inv, acc, pref[i]);             // 1 / den_i
        curve25519_mul(acc, acc, den[i]);              // strip den_i
        curve25519_mul(y, num[i], inv);                // y_i = num_i / den_i
        check_and_record(y, start_unit + (unsigned long long)i);
    }
}

// -------------------------------------------------------------------------
// Host-callable single-key pack: full compressed pubkey (incl. parity) for a
// given clamped scalar. Used to verify/display hits.
// -------------------------------------------------------------------------
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

// Per-thread local memory (bytes) the given window instantiation needs.
static size_t window_local_bytes(int window) {
    cudaFuncAttributes fa;
    cudaError_t e;
    switch (window) {
        case 1024: e = cudaFuncGetAttributes(&fa, vanity_kernel<1024>); break;
        case 512:  e = cudaFuncGetAttributes(&fa, vanity_kernel<512>);  break;
        case 256:  e = cudaFuncGetAttributes(&fa, vanity_kernel<256>);  break;
        case 128:  e = cudaFuncGetAttributes(&fa, vanity_kernel<128>);  break;
        default:   e = cudaFuncGetAttributes(&fa, vanity_kernel<64>);   break;
    }
    return e == cudaSuccess ? fa.localSizeBytes : (size_t)3 * window * 40;
}

// Largest supported window whose worst-case local-memory reserve
// (localBytes * maxResidentThreads) fits in free VRAM with margin.
static int auto_window(int tpb) {
    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);
    int numSM = 1, maxThreadsSM = 1024, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&maxThreadsSM, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    for (int w : kWindows) {
        size_t resident = (size_t)numSM * maxThreadsSM;   // worst case the driver reserves for
        size_t reserve = resident * window_local_bytes(w);
        if ((double)reserve < 0.85 * (double)freeB) return w;
    }
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

int main(int argc, char **argv) {
    std::string prefix;
    long limit = 1;
    int blocks = 512;
    int tpb = 256;
    int device = 0;
    int window = 0;              // 0 = auto-fit to VRAM
    bool progress = true;
    bool selftest = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char *n) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", n); exit(1); }
            return argv[++i];
        };
        if (a == "--selftest") selftest = true;
        else if (a == "-l" || a == "--limit") limit = parse_long(need("--limit"));
        else if (a == "--blocks") blocks = (int)parse_long(need("--blocks"));
        else if (a == "--tpb") tpb = (int)parse_long(need("--tpb"));
        else if (a == "-d" || a == "--device") device = (int)parse_long(need("--device"));
        else if (a == "-w" || a == "--window") window = (int)parse_long(need("--window"));
        else if (a == "--no-progress") progress = false;
        else if (a == "-h" || a == "--help") {
            printf("Usage: %s <HEX_PREFIX> [options]\n"
                   "  -l, --limit N     stop after N matches (0 = infinite) [1]\n"
                   "  -w, --window N    batch size/thread: 64|128|256|512|1024 [auto-fit VRAM]\n"
                   "                    bigger = faster but more GPU memory\n"
                   "      --blocks N    CUDA blocks [512]\n"
                   "      --tpb N       threads per block [256]\n"
                   "  -d, --device I    CUDA device index [0]\n"
                   "      --no-progress suppress progress output\n"
                   "      --selftest    run correctness self-tests and exit\n",
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

    if (prefix.empty() || prefix.size() > 64) {
        fprintf(stderr, "Prefix must be 1-64 hex characters. Use --help.\n");
        return 1;
    }

    // Resolve the window: explicit --window (snapped to a supported size) or
    // auto-fit to free VRAM.
    if (window == 0) {
        window = auto_window(tpb);
        fprintf(stderr, "Window: auto-selected %d (fits GPU memory)\n", window);
    } else {
        int snap = nearest_window(window);
        if (snap != window)
            fprintf(stderr, "Window: %d not supported, using %d (supported: 64/128/256/512/1024)\n",
                    window, snap);
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

    // Shared precomputed step table i*D (affine x, y and x*y), i in [1,window).
    bignum25519 *d_gx, *d_gy, *d_gp;
    cuda_check(cudaMalloc(&d_gx, sizeof(bignum25519) * window), "malloc gx");
    cuda_check(cudaMalloc(&d_gy, sizeof(bignum25519) * window), "malloc gy");
    cuda_check(cudaMalloc(&d_gp, sizeof(bignum25519) * window), "malloc gp");
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

    return fails == 0 ? 0 : 2;
}
