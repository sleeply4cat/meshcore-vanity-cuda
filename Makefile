# meshcore-vanity-cuda
NVCC    ?= nvcc
ARCH    ?= sm_86
# Codegen target(s). Default: a single native arch. Override GENCODE to build a
# fat binary covering many GPUs (see `make release`), e.g.
#   make GENCODE="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86"
GENCODE ?= -arch=$(ARCH)
# The batch window is chosen at runtime (--window, or auto-fit to VRAM); no
# compile-time knob needed. Integer math only (no --use_fast_math).
NVFLAGS := -O3 $(GENCODE) -lineinfo -Xptxas -O3 --std=c++14

BIN := meshcore-vanity

all: $(BIN)

$(BIN): src/main.cu src/ed25519.cuh
	$(NVCC) $(NVFLAGS) src/main.cu -o $@

# Portable multi-arch build for distribution: SASS for common GPUs plus PTX
# fallbacks (JIT-compiled by the driver on newer/unlisted GPUs). A GPU without
# its own SASS pays for that JIT on the first run — over 3 minutes for this
# kernel set — so every consumer generation gets SASS. sm_120 (RTX 50) needs
# CUDA >= 12.8; CUDA 13 dropped sm_60/sm_70, so stay on 12.x. Expect ~50 min
# of single-core compile (sm_60 alone ~13): `nvcc --threads` would build the
# architectures in parallel, but with CUDA 12.8 it reproducibly fails at the
# device-link step ("nvlink fatal: Could not read file ..._dlink.reg.c").
# Keep in sync with GENCODE in .github/workflows/build.yml.
release:
	$(NVCC) -O3 -Xptxas -O3 --std=c++14 \
	  -gencode arch=compute_60,code=sm_60 \
	  -gencode arch=compute_70,code=sm_70 \
	  -gencode arch=compute_75,code=sm_75 \
	  -gencode arch=compute_80,code=sm_80 \
	  -gencode arch=compute_86,code=sm_86 \
	  -gencode arch=compute_89,code=sm_89 \
	  -gencode arch=compute_90,code=sm_90 \
	  -gencode arch=compute_90,code=compute_90 \
	  -gencode arch=compute_120,code=sm_120 \
	  -gencode arch=compute_120,code=compute_120 \
	  src/main.cu -o $(BIN)

selftest: $(BIN)
	./$(BIN) --selftest

clean:
	rm -f $(BIN) build.env

.PHONY: all release clean selftest
