# Build the spike tools on the GPU machine.
#   make OGKM=/path/to/open-gpu-kernel-modules
# OGKM must be checked out at the same version as the running driver.

OGKM ?= $(HOME)/open-gpu-kernel-modules
NVCC ?= nvcc
INC = -I$(OGKM)/src/common/sdk/nvidia/inc \
      -I$(OGKM)/src/nvidia/arch/nvalloc/unix/include \
      -I$(OGKM)/kernel-open/common/inc

all: workloads/w1_busy workloads/w2_work workloads/hp_probe shim/libdynagpu_shim.so probe/agent_probe

workloads/w1_busy: workloads/w1_busy.cu
	$(NVCC) -O2 -o $@ $<

workloads/w2_work: workloads/w2_work.cu
	$(NVCC) -O2 -o $@ $<

workloads/hp_probe: workloads/hp_probe.cu
	$(NVCC) -O2 -o $@ $<

shim/libdynagpu_shim.so: shim/dynagpu_shim.c
	$(CC) -O2 -Wall -fPIC -shared $(INC) -o $@ $< -ldl -lpthread

probe/agent_probe: probe/agent_probe.c
	$(CC) -O2 -Wall $(INC) -o $@ $<

clean:
	rm -f workloads/w1_busy workloads/w2_work workloads/hp_probe shim/libdynagpu_shim.so probe/agent_probe
