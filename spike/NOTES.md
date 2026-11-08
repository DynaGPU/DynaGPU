# Spike lab notebook

Newest first. Every entry records the GPU, the driver and the exact commands.

## 2026-11-08 (afternoon): HAMi-core comparison, 1 × H100 (driver 610.57.04)

**Machine.** No L40S capacity on Nebius (1-GPU, the larger 1-GPU and the 2-GPU presets all returned `NotEnoughResources`). Switched to `gpu-h100-sxm` / `1gpu-16vcpu-200gb`; it came up first try. First run on Hopper: the shim needed no changes (GPFIFO class `0xc86f`, 4 groups per context).

**Getting HAMi-core to limit anything** (commit `1721252`) took four tries:
1. A CMake Release build linked only with `LIBRARY_PATH=/usr/local/cuda/lib64/stubs` (for `-lnvidia-ml`). It loaded, printed its messages, and changed nothing: 12,753 vs 12,757 kps at a 40% limit.
2. Logs showed no `Into cuInit`. I suspected CUDA 13 and built W2 with CUDA 12.8, static and shared. Still nothing. **That suspicion was wrong.**
3. Read `libvgpu.c`: its `dlsym` hook looks up its own functions in `/usr/local/vgpu/libvgpu.so` unless `CUDA_REDIRECT` is set. Set it. Still nothing.
4. `nm -D` showed the Release build exports only `dlsym`: `CMakeLists.txt` adds `-fvisibility=hidden` for non-Debug builds. Rebuilt with `build.sh` (Debug): 208 `cu*` exports, and a 40% limit gave 41–44% on all four builds.

**Comparison** (`run_hami.py`, about 40 min): see `results/hami-comparison.md`. HAMi-core: graphs 50/50 (no effect), 40% cap on graphs → 100%, long kernels → 52.8% split and 66.1% cap; short kernels fine but noisy. DynaGPU: 70.1% / 40.2–40.5% everywhere. Launch +33% with a HAMi-core limit, 0 with DynaGPU.

**Bug (DynaGPU): pause failed with both loaded.** 1,981 of 2,000 calls returned `0x21`. With HAMi-core, the app makes 3 contexts (2 thrown away at start-up). Reproduced without HAMi-core: `cudaDeviceReset()` twice before work → pause fails. Fix part 1: watch `NV_ESC_RM_FREE` and try every group. Still failing; a new `dump` command showed the live group carrying 4 channels of a freed group, because `tsgs[ntsg++]` reused a slot without clearing `nchan`. Fix part 2: `memset` the slot. After both: resets 0/2/5 all pause, 51 of 2,000 calls fail and all of them at start-up, H100 quick checks pass.

**Both limiting compute at once** (HAMi-core 70/30 + DynaGPU 7000/3000): 61% in one run, 75% in another. Don't.

**Cleanup.** `terraform destroy`: 2 resources destroyed; 0 instances, 0 disks. About 60 minutes of H100 time.

## 2026-11-08: week 3 (E9–E12), 2 × L40S, drivers 610 / 580 / 570 / 535 (open)

**Machine.** Terraform with `-var preset=2gpu-64vcpu-384gb` (E11 needs 2 GPUs). It came up first try. PyTorch 2.15.0+cu132 with NCCL 2.30.7.

**Shim changes.** Lease (`DYNAGPU_LEASE_MS`, watchdog every 50 ms: resume + restore CUDA's 2048 µs turn length), `hb` command, `%p` in the socket path.

**Bug: pause failed with `0x21` on 2 GPUs.** The log showed two subdevices under one client (`hDevice` `0x5c000002` and `0x5c00000a`). CUDA makes a device + subdevice per GPU, and the shim took the last (GPU 1's) for an app on GPU 0. Fixed: match client **and** parent device. After that, a lease smoke test resumed a paused app 1.03 s after the agent went quiet.

**Runner bugs (mine).** A stale socket file from the previous rep made `sendto` fail with `ECONNREFUSED`, and the agent crashed on the final resume to an exited app. Fixed both; restarted from the saved calibration. Separately, NCCL ranks' output lines got glued together, so each line is now one `os.write`.

**Full run** (`run_week3.py`, about 30 min), then gang + clock follow-ups (about 5 min):
- E9 alone: +1.2 to +4.3 pts over the cap. Busy neighbour: −0.3 / −1.1. Overshoot is about 6% of the cap at 20–60% and 2.5% at 80%. Hypothesis: clock boost. Clock check: flat out 2,307 MHz at the 350 W limit (reason `0x4`); 40% cap 2,460 MHz (+6.6%); 80% cap 2,389 MHz (+3.5%). Corrected: 39.8% and 79.2%. Confirmed.
- NVML `pmon`: right ratio for steady weighted apps, wrong absolute values, unusable with pausing.
- E10: resume 1.048–1.049 s (5/5); 9:1 → 50.0% after 1.042–1.045 s (3/3); 15/15 long pauses held with heartbeats, 0 false expiries.
- E11: 0 errors, 0 hangs over baseline, spin test and 600 pause cycles. One rank paused → the other GPU 100% busy (spinning). Both paused → both 0%.
- **Surprise:** each GPU shared 7:3 → all-reduce at 34.8%, not 70%. Agent slots aligned across GPUs: 69.8%. Offset 50 ms: 41.8% (overlap 40%). Gang scheduling is needed.

**E12 static.** Blobless clone of `open-gpu-kernel-modules`; checked out headers for the newest tag of 15 branches (525–620). Everything the shim uses is identical from 535 to 620. Flag values looked different until I read `control.h`: NVIDIA renumbered the bits in 555 (`NON_PRIVILEGED` `0x10` → `0x8`). Decoded, every control is non-root in all 15.

**E12 runtime.** The Nebius image pins every NVIDIA package (`/etc/apt/preferences.d`). 580 via the CUDA apt repo needed a full purge first (package names differ: `libnvidia-cfg1` vs `libnvidia-cfg1-580`), then a reboot. Ubuntu's `nvidia-driver-570-open` turned out to install 580. 570.211.01 and 535.309.01 went in with NVIDIA's `.run` installer after `rmmod gdrdrv` (it holds `nvidia`). Workloads for them were built with CUDA 12.8 for `sm_89`. All three passed: 70.1% split, 1,000 pauses, checksum match. The first pause on 535 took 6.3 ms.

**Design changes.** Caps are a share of GPU time. The agent does its own per-app accounting. Gang scheduling for multi-GPU pods. Match subdevices by GPU. One shim build for 535–620.

**Cleanup.** `terraform destroy`: 2 resources destroyed; 0 instances, 0 disks. About 75 minutes of 2 × L40S time.

**Next.** The spike is done (GO). Open: the HAMi-core comparison (needs approval), the slow-call tail, the E9 long-kernel residual, gang scheduling with many GPUs, H100 / NVLink / MIG.

## 2026-11-07: week 2 (E5–E8), L40S, driver 610.57.04 (open)

**Machine.** Same Terraform, now defaulting to `gpu-l40s-d`. It came up first try. Fresh PyTorch install gave 2.15.0+cu132 (week 1: 2.11).

**New workload W2.** W1's spin kernels count clock time, so a paused long kernel would count its paused time as work. W2 does fixed integer work per thread and keeps state in GPU memory. Kernel lengths: `--iters 4000` ≈ 67 µs, 40,000 ≈ 0.66 ms, 2,000,000 ≈ 33 ms.

**Pause, first try: failed silently.** `GPFIFO_SCHEDULE(bEnable=false)` + `PREEMPT` on the channel group returned `NV_OK`, but the app kept full speed through a 5 s pause. Read `kchangrpapiCtrlCmdGpFifoSchedule_IMPL`: on GSP clients it is forwarded to the firmware via RPC.

**Pause, second try: works.** `NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS` (flags `0x10108`, non-privileged) on the subdevice, listing the group's channels. The shim now records subdevices (`0x2080`) and GPFIFO channels (class `0xc56f`; 8 per compute group). A 5 s pause cut throughput by exactly 5 s' worth. Pause ≈ 0.4 ms, resume ≈ 0.3 ms.

**Bug: second app failed with `0x23 INVALID_CLIENT`.** Its subdevice was under client `0xc1d000a2`, its channel group under `0xc1d000a6`. CUDA had made a short-lived client first, and the shim kept the first subdevice it saw. Fixed: one subdevice per client, matched to the group's client. After the fix 3 apps paused cleanly, with no NVRM messages.

**HAMi-core.** Cloned (`1721252`). The full build failed at its test programs. Building only the library was blocked by the session's safety policy (outside code). Skipped; needs user approval.

**Full run** (`run_week2.py`, about 40 min), then the E7 sweep (about 5 min):
- E5: all 7:3 cases within 0.7 pts, including graphs and 33 ms kernels.
- PyTorch: 66.7% (sync every 16), 69.7% (256), 70.4% (never idle). Week-1 hypothesis confirmed.
- E6: p99 4.47 ms default → 1.00 / 1.02 ms with pausing (+962 / +985 µs vs alone; pass mark 1000). Interleave HIGH: no effect. LP turns 1 ms: 2.33 ms.
- E7: loss 10.6% / 12.2% / 13.4% at 1 ms (2 / 4 / 8 apps) → fail. Sweep: 2 apps ≤5% from 2 ms; 4 apps from 5 ms. Launch 1616 vs 1631 ns, round trip identical.
- E8: 2 × 10,000 cycles, checksums identical, 0 Xid, no hang. 3 of 40,000 calls took 12–190 ms (unexplained).

**Design changes.** Minimum turn 5 ms. Priority by pausing. Pause with `DISABLE_CHANNELS`. Subdevice per client. Leases must allow for about 200 ms control calls.

**Cleanup.** `terraform destroy`: 2 resources destroyed, none left. About 70 minutes of GPU time.

**Next (week 3).** E9 (hard caps), E10 (agent dies → lease), E11 (NCCL; needs 2 GPUs), E12 (driver 535–610). Also: the HAMi-core comparison if approved, the slow-call tail, and the 1 ms vs 2 ms jump.

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
