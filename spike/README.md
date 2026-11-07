# DynaGPU feasibility spike

Code and results for the feasibility tests (E1–E12) that decide whether DynaGPU's approach works: weighted GPU time-sharing through the GPU's own channel-group timeslices.

**Status:** week 1 (E1–E4) passed on 2026-11-07. See [`results/week1.md`](results/week1.md).

## Layout

| Path | What it is |
| --- | --- |
| [`shim/dynagpu_shim.c`](shim/dynagpu_shim.c) | `LD_PRELOAD` library. Finds each app's channel groups (E1) and sets the compute group's timeslice from inside the app (E2 route A) |
| [`probe/agent_probe.c`](probe/agent_probe.c) | Tries to set another process's timeslice from outside (E2 routes B and C) |
| [`workloads/w1_busy.cu`](workloads/w1_busy.cu) | W1: fixed-size spin kernels kept queued back to back; counts kernels per second |
| [`workloads/torch_matmul.py`](workloads/torch_matmul.py) | PyTorch matrix-multiply loop for E1 and an E3 cross-check |
| [`scripts/run_spike.py`](scripts/run_spike.py) | Runs solo, default-pair, E3 and E4 tests; writes CSVs |
| [`scripts/run_extra.py`](scripts/run_extra.py) | Runtime timeslice change and PyTorch checks |
| [`scripts/plot_e3.py`](scripts/plot_e3.py) | Draws the E3 chart as SVG (no dependencies) |
| [`infra/`](infra/) | Terraform for one GPU VM on Nebius |
| [`results/`](results/) | Reports and raw data, one folder per run |
| [`NOTES.md`](NOTES.md) | Lab notebook |

## Requirements

- Linux with an NVIDIA GPU (Pascal or newer) and a real driver. **WSL2 does not work**: it has no `/dev/nvidiactl`.
- The NVIDIA driver, preferably the open kernel module. Check with `cat /proc/driver/nvidia/version`.
- The CUDA toolkit (`nvcc`), `gcc`, `make`, Python 3.
- NVIDIA's `open-gpu-kernel-modules` source at **exactly the same version** as the running driver, for its headers (MIT licensed). Nothing from it is copied into this repo.

## Run it

```bash
# 1. A GPU machine (optional: Nebius via Terraform)
cd spike/infra
cp terraform.tfvars.example terraform.tfvars   # fill in project_id and subnet_id
terraform init && terraform apply
# Stop the GPU at the end of a day without deleting:  terraform apply -var stopped=true
# Delete everything when finished:                   terraform destroy

# 2. On the GPU machine
git clone --depth 1 --branch "$(cat /sys/module/nvidia/version)" \
    https://github.com/NVIDIA/open-gpu-kernel-modules.git ~/open-gpu-kernel-modules
cd DynaGPU/spike
make OGKM=$HOME/open-gpu-kernel-modules

# E1 + E2 route A: see the channel groups and set a timeslice
DYNAGPU_TIMESLICE_US=7000 DYNAGPU_LOG=/tmp/shim.log \
  LD_PRELOAD=$PWD/shim/libdynagpu_shim.so ./workloads/w1_busy --seconds 5
cat /tmp/shim.log

# E2 routes B and C
DYNAGPU_STATE=/tmp/state.txt LD_PRELOAD=$PWD/shim/libdynagpu_shim.so \
  ./workloads/w1_busy --seconds 15 &
sleep 4; awk '$5==1' /tmp/state.txt          # pid fd hClient hTsg engineType
./probe/agent_probe <hClient> <hTsg> 5000

# E3, E4 and overhead (about 25 minutes)
python3 scripts/run_spike.py --bin $PWD/workloads/w1_busy \
  --shim $PWD/shim/libdynagpu_shim.so --out $PWD/results/run1
```

## Shim settings

| Variable | Meaning |
| --- | --- |
| `DYNAGPU_TIMESLICE_US` | Timeslice for the app's compute channel group. Also replaces CUDA's own 2048 µs setting |
| `DYNAGPU_CTL` | File the shim checks every 50 ms. When it holds a new number, that becomes the timeslice of the running app |
| `DYNAGPU_STATE` | File where the shim appends `pid fd hClient hTsg engineType` for each channel group |
| `DYNAGPU_LOG` | Log file (default: stderr) |

## Notes on Nebius

- On 2026-11-07, `gpu-l40s-a` returned "Not enough resources". `gpu-l40s-d` with `1gpu-16vcpu-96gb` worked and is now the default in `infra/main.tf`.
- Login uses the Nebius CLI profile (federation). The first `terraform apply` may open a browser.
- `terraform.tfvars` holds your account IDs and is not committed.
