#!/usr/bin/env python3
"""E12 (static part): is the driver interface the shim uses the same across
NVIDIA driver versions?

For every version of NVIDIA's open-gpu-kernel-modules (MIT) given, this
  1. compiles a small C program against that version's headers and prints
     every ioctl number, class ID, control ID, structure size and field offset
     the shim depends on;
  2. reads the privilege flags of each control from the generated
     resource-manager tables (0x8 = callable without root);
  3. checks that the shim itself compiles against those headers.

Usage: e12_headers.py --trees DIR --shim shim/dynagpu_shim.c --out DIR
       DIR/<version>/ holds that version's checkout (only these paths are needed:
       src/common/sdk/nvidia/inc, src/nvidia/arch/nvalloc/unix/include,
       kernel-open/common/inc, src/nvidia/generated/g_kernel_channel_group_api_nvoc.c,
       src/nvidia/generated/g_subdevice_nvoc.c)
"""
import argparse
import csv
import os
import re
import subprocess
import tempfile

P = argparse.ArgumentParser()
P.add_argument("--trees", required=True)
P.add_argument("--shim", required=True)
P.add_argument("--out", required=True)
A = P.parse_args()

PROBE = r"""
#include <stddef.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include "nvtypes.h"
#include "nvos.h"
#include "nvstatus.h"
#include "nv_escape.h"
#include "nv-ioctl-numbers.h"
#include "class/cla06c.h"
#include "alloc/alloc_channel.h"
#include "class/cl2080.h"
#include "ctrl/ctrl2080/ctrl2080fifo.h"
#include "ctrl/ctrla06c.h"
#define V(x) printf("%s=0x%llx\n", #x, (unsigned long long)(x))
#define S(t) printf("sizeof(%s)=%zu\n", #t, sizeof(t))
#define O(t, f) printf("offsetof(%s.%s)=%zu\n", #t, #f, offsetof(t, f))
int main(void) {
  V(NV_IOCTL_MAGIC); V(NV_IOCTL_BASE); V(NV_ESC_REGISTER_FD);
  V(NV_ESC_RM_ALLOC); V(NV_ESC_RM_CONTROL); V(NV_ESC_RM_DUP_OBJECT);
  V(KEPLER_CHANNEL_GROUP_A); V(NV20_SUBDEVICE_0); V(NV2080_ENGINE_TYPE_GRAPHICS);
  V(NVA06C_CTRL_CMD_GPFIFO_SCHEDULE); V(NVA06C_CTRL_CMD_SET_TIMESLICE);
  V(NVA06C_CTRL_CMD_GET_TIMESLICE); V(NVA06C_CTRL_CMD_PREEMPT);
  V(NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL);
  V(NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS); V(NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES);
  V(NV_OK); V(NV_ERR_INVALID_CLIENT); V(NV_ERR_INSUFFICIENT_PERMISSIONS);
  S(NVOS21_PARAMETERS); O(NVOS21_PARAMETERS, hRoot); O(NVOS21_PARAMETERS, hObjectParent);
  O(NVOS21_PARAMETERS, hObjectNew); O(NVOS21_PARAMETERS, hClass);
  O(NVOS21_PARAMETERS, pAllocParms); O(NVOS21_PARAMETERS, status);
  S(NVOS64_PARAMETERS); O(NVOS64_PARAMETERS, status);
  S(NVOS54_PARAMETERS); O(NVOS54_PARAMETERS, hClient); O(NVOS54_PARAMETERS, hObject);
  O(NVOS54_PARAMETERS, cmd); O(NVOS54_PARAMETERS, params); O(NVOS54_PARAMETERS, paramsSize);
  O(NVOS54_PARAMETERS, status);
  S(NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS);
  O(NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS, engineType);
  S(NVA06C_CTRL_TIMESLICE_PARAMS); S(NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS);
  S(NVA06C_CTRL_PREEMPT_PARAMS); S(NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS);
  S(NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS);
  O(NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS, bDisable);
  O(NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS, numChannels);
  O(NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS, bOnlyDisableScheduling);
  O(NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS, hClientList);
  O(NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS, hChannelList);
  printf("ioctl(RM_CONTROL)=0x%lx\n", (unsigned long)_IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_CONTROL, NVOS54_PARAMETERS));
  return 0;
}
"""

CONTROLS = {  # control -> generated file that holds its export table
    "SET_TIMESLICE 0xa06c0103": ("g_kernel_channel_group_api_nvoc.c", 0xa06c0103),
    "GET_TIMESLICE 0xa06c0104": ("g_kernel_channel_group_api_nvoc.c", 0xa06c0104),
    "GPFIFO_SCHEDULE 0xa06c0101": ("g_kernel_channel_group_api_nvoc.c", 0xa06c0101),
    "PREEMPT 0xa06c0105": ("g_kernel_channel_group_api_nvoc.c", 0xa06c0105),
    "SET_INTERLEAVE_LEVEL 0xa06c0107": ("g_kernel_channel_group_api_nvoc.c", 0xa06c0107),
    "FIFO_DISABLE_CHANNELS 0x2080110b": ("g_subdevice_nvoc.c", 0x2080110b),
}


def incs(tree):
    return ["-I", f"{tree}/src/common/sdk/nvidia/inc", "-I",
            f"{tree}/src/nvidia/arch/nvalloc/unix/include", "-I", f"{tree}/kernel-open/common/inc"]


def flags_for(tree, fname, method):
    """Find the export-table entry for `method` and return its flags."""
    path = f"{tree}/src/nvidia/generated/{fname}"
    if not os.path.exists(path):
        return "file missing"
    src = open(path).read()
    m = re.search(r"methodId=\*/\s*0x%xu" % method, src)
    if not m:
        return "not found"
    before = src[:m.start()]
    fl = re.findall(r"flags=\*/\s*(0x[0-9a-f]+)u", before)
    return fl[-1] if fl else "?"


def vkey(v):
    return [int(x) for x in v.split(".")]


versions = sorted(os.listdir(A.trees), key=vkey)
rows, compiled = {}, {}
tmp = tempfile.mkdtemp()
open(f"{tmp}/probe.c", "w").write(PROBE)
for v in versions:
    tree = os.path.join(A.trees, v)
    exe = f"{tmp}/probe-{v}"
    r = subprocess.run(["gcc", "-w", "-o", exe, f"{tmp}/probe.c"] + incs(tree),
                       capture_output=True, text=True)
    vals = {}
    if r.returncode == 0:
        for line in subprocess.run([exe], capture_output=True, text=True).stdout.splitlines():
            k, val = line.split("=", 1)
            vals[k] = val
    else:
        vals["probe build"] = "FAILED: " + r.stderr.strip().splitlines()[0][:120]
    for name, (fname, method) in CONTROLS.items():
        vals[f"flags {name}"] = flags_for(tree, fname, method)
    s = subprocess.run(["gcc", "-fsyntax-only", "-Wall", A.shim] + incs(tree),
                       capture_output=True, text=True)
    vals["shim compiles"] = "yes" if s.returncode == 0 else "NO: " + s.stderr.strip()[:120]
    rows[v] = vals

keys = []
for v in versions:
    for k in rows[v]:
        if k not in keys:
            keys.append(k)
os.makedirs(A.out, exist_ok=True)
with open(os.path.join(A.out, "e12_interface.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item"] + versions + ["same in all"])
    for k in keys:
        vals = [rows[v].get(k, "-") for v in versions]
        w.writerow([k] + vals + ["yes" if len(set(vals)) == 1 else "NO"])

diff = [k for k in keys if len({rows[v].get(k, "-") for v in versions}) > 1]
print(f"versions: {' '.join(versions)}")
print(f"items checked: {len(keys)}; differing: {len(diff)}")
for k in diff:
    print(f"  {k}: " + ", ".join(f"{v}={rows[v].get(k, '-')}" for v in versions))
