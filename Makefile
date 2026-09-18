# Два независимых пути сборки:
#   make cpu      CPU-референс + тесты, обычный C++17, CUDA не нужна
#   make gpu      решатель на CUDA (нужен nvcc)
#   make          оба, если nvcc найден, иначе только cpu

CXX      ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Iinclude

NVCC     ?= nvcc
# Должна соответствовать карте, иначе драйвер каждый запуск JIT-компилирует из PTX, и измеряется не тот код, что собран.
# sm_70 Volta, sm_75 Turing, sm_80 A100, sm_86 Ampere GeForce, sm_89 Ada, sm_90 Hopper.
GPU_ARCH ?= sm_70
NVCCFLAGS ?= -O3 -std=c++17 -Iinclude --generate-code arch=compute_$(subst sm_,,$(GPU_ARCH)),code=$(GPU_ARCH)

BIN      := bin
HAVE_NVCC := $(shell command -v $(NVCC) 2>/dev/null)

CPU_SOURCES := src/cc_graph.cpp src/cc_pbils_cpu.cpp

.PHONY: all cpu gpu test clean
ifeq ($(HAVE_NVCC),)
all: cpu
	@echo "note: nvcc not found, GPU target skipped (run 'make gpu' on a CUDA host)"
else
all: cpu gpu
endif

cpu: $(BIN)/cc_cpu $(BIN)/cc_verify

$(BIN):
	@mkdir -p $(BIN)

$(BIN)/cc_cpu: $(CPU_SOURCES) src/main_cpu.cpp include/*.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $(CPU_SOURCES) src/main_cpu.cpp

$(BIN)/cc_verify: $(CPU_SOURCES) src/cc_verify.cpp include/*.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $(CPU_SOURCES) src/cc_verify.cpp

gpu: $(BIN)/cc_gpu

$(BIN)/cc_gpu: $(CPU_SOURCES) cuda/cc_pbils_gpu.cu cuda/main_gpu.cu include/*.hpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ cuda/cc_pbils_gpu.cu cuda/main_gpu.cu $(CPU_SOURCES)

test: $(BIN)/cc_verify
	./$(BIN)/cc_verify

clean:
	rm -rf $(BIN)
