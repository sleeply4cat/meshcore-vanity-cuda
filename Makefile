# meshcore-vanity-cuda
NVCC    ?= nvcc
ARCH    ?= sm_86
# Codegen target(s). Default: a single native arch. Override GENCODE to build a
# fat binary covering many GPUs (see `make release`), e.g.
#   make GENCODE="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86"
GENCODE ?= -arch=$(ARCH)
# The batch window is chosen at runtime (--window, or auto-fit to VRAM); no
# compile-time knob needed. Integer math only (no --use_fast_math).
# EXTRA: extra nvcc flags, e.g. EXTRA=-DVANITY_FULL_FINAL_MUL for an A/B build
# that forms the final products in full instead of filtering on part of them.
EXTRA   ?=
NVFLAGS := -O3 $(GENCODE) -lineinfo -Xptxas -O3 --std=c++14 $(EXTRA)
# std::thread (the JIT notice); a no-op with glibc >= 2.34.
LDLIBS  := -Xcompiler -pthread

BIN := meshcore-vanity

all: $(BIN)

$(BIN): src/main.cu src/ed25519.cuh
	$(NVCC) $(NVFLAGS) src/main.cu -o $@ $(LDLIBS)

# Release builds for distribution, two variants of the same program:
#
# `make release` (full): machine code (SASS) for every consumer generation plus
# PTX fallbacks (JIT-compiled by the driver on newer/unlisted GPUs), so no GPU
# of the listed generations waits for the driver's compiler on its first run.
# sm_120 (RTX 50) needs CUDA >= 12.8; CUDA 13 dropped sm_60/sm_70, so stay on
# 12.x. Expect ~50 min of single-core compile (sm_60 alone ~13): `nvcc
# --threads` would build the architectures in parallel, but with CUDA 12.8 it
# reproducibly fails at the device-link step ("nvlink fatal: Could not read
# file ..._dlink.reg.c").
#
# `make release-slim`: PTX only, no machine code at all, a fraction of the
# size and build time. The driver compiles it for whatever GPU runs it on the
# first run (4-5 min and ~2.5 GB of RAM on a laptop; the program says so while
# it waits) and caches the result. Needs a driver at least as new as the
# toolkit that built it (R575+ for CUDA 12.9), since that is what compiles it.
#
# Keep in sync with GENCODE_FULL / GENCODE_SLIM in .github/workflows/build.yml.
RELEASE_FLAGS := -O3 -Xptxas -O3 --std=c++14
GENCODE_FULL := \
  -gencode arch=compute_60,code=sm_60 \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90 \
  -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_120,code=compute_120
GENCODE_SLIM := -gencode arch=compute_60,code=compute_60

release:
	$(NVCC) $(RELEASE_FLAGS) $(GENCODE_FULL) src/main.cu -o $(BIN) $(LDLIBS)

release-slim:
	$(NVCC) $(RELEASE_FLAGS) $(GENCODE_SLIM) src/main.cu -o $(BIN) $(LDLIBS)

selftest: $(BIN)
	./$(BIN) --selftest

clean:
	rm -f $(BIN) build.env

.PHONY: all release release-slim clean selftest
