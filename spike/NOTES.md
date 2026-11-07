# Spike lab notebook

Newest first. Every entry records the GPU, the driver and the exact commands.

## 2026-11-07: week 1 (E1–E4), L40S, driver 610.57.04 (open)

**Machine.** Terraform on Nebius. The first try with `gpu-l40s-a` / `1gpu-8vcpu-32gb` failed after about 5 minutes: `NotEnoughResources: VM schedule timeout`. Retried with `gpu-l40s-d` / `1gpu-16vcpu-96gb`; it came up in 28 s. The image `ubuntu24.04-cuda13-latest` already had driver 610.57.04 as the open kernel module, plus CUDA 13.3. Cloned `open-gpu-kernel-modules` at tag `610.57.04`.

**Before testing.** Read `ctrla06c.h`. The control IDs match the design doc: `SET_TIMESLICE 0xa06c0103`, `GET_TIMESLICE 0xa06c0104`, `GPFIFO_SCHEDULE 0xa06c0101`, `PREEMPT 0xa06c0105`. The doc comment says the value is rounded down to a valid hardware value and takes effect immediately. In `g_kernel_channel_group_api_nvoc.c`, `SET_TIMESLICE` has flags `0x10008`, which includes `RMCTRL_FLAGS_NON_PRIVILEGED` (`0x8`).

**E1.** First shim run with W1: 3 channel groups per context, engine types 1, 11 and 12 (`GR0`, `COPY2`, `COPY3`). The log also showed CUDA itself calling `0xa06c0103` on the compute group. A second run showed the value is **2048 µs**. Changed the shim: (1) only set the compute group (engine 1); (2) catch CUDA's own `SET_TIMESLICE` and replace the value. After that: `SET OK`, and `GET` returns 7000. PyTorch 2.11: 1 process, 1 compute group, set to 7000. Pass.

**E2.** Route A (in-process) works. Route B gives `0x23 INVALID_CLIENT`. Route C: own client + `REGISTER_FD` + device all OK; `DUP_OBJECT` gives `0x1b INSUFFICIENT_PERMISSIONS`. Same as root. No `dmesg` messages. Pass.

**E3 / E4 / overhead.** `run_spike.py`, 5 reps, 20 s runs with 4 s warm-up, about 25 minutes. Every split within 0.8 points of target, and runs repeat to 3 decimals. E4: 95.6% (bursty neighbour) and 99.7% (idle neighbour). Shim overhead 0.07%, inside the noise. Pass.

**Extras.** Changing the timeslice while the apps run (control file): 69.8% vs 70%. PyTorch 7:3: 66.7%, which is 3.3 points short. Likely because the script syncs every 16 matmuls, gives up its turn while idle, and B uses the time (not yet confirmed). Week-2 item.

**Surprises.**
- CUDA sets its own timeslice (2048 µs) after creating the channel group.
- Shorter cycles cost real throughput: 94.1% at 2 ms, 96.5% at CUDA's default 4.1 ms, 97.9% at 10 ms.
- Outside routes are refused even as root, so the agent must work through the shim.

**Cleanup.** `terraform destroy`: 2 resources destroyed; no instances or disks left. About 45 minutes of GPU time.

**Next (week 2).** E5 (CUDA graphs, long kernels, vLLM), E6 (preemption latency), E7 (cost), E8 (10,000 pause cycles). Also investigate the PyTorch shortfall.
