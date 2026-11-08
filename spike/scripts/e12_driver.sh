#!/usr/bin/env bash
# E12 (runtime part): switch a THROWAWAY GPU test machine to another NVIDIA
# open driver, then rebuild and re-run the quick checks against it.
#
#   e12_driver.sh install-run <version>          NVIDIA's .run installer (used for 570, 535)
#   e12_driver.sh install-apt <version> <deb>    NVIDIA's CUDA apt repo (used for 580), then reboot
#   e12_driver.sh test <version> [nvcc]          rebuild shim + W2, run run_week3.py --only quick
#
# Drivers older than 580 can't run CUDA 13 programs: pass a CUDA 12 nvcc to
# `test` (e.g. /usr/local/cuda-12.8/bin/nvcc). CUDA 12 programs compiled for
# the GPU (sm_89, no PTX JIT) also run on 535 through minor-version compatibility.
#
# The Nebius image locks all NVIDIA packages to its driver (/etc/apt/preferences.d).
# install-apt ignores those locks for its own apt calls only, keeping the kernel lock.
set -euo pipefail
cmd=$1; ver=$2
SPIKE=$(cd "$(dirname "$0")/.." && pwd)

purge_apt_driver() {
  # Package names differ between branches (libnvidia-cfg1 in 610, libnvidia-cfg1-580
  # in 580), so remove the current driver completely first.
  local p
  p=$(dpkg -l | awk '/^(ii|iU|iF|rc)/{print $2}' | grep -E '^(nvidia-(open|driver|dkms|kernel|firmware|imex|fabricmanager|modprobe|persistenced|utils|compute-utils)|libnvidia-(cfg|compute|decode|encode|fbc|gl|extra|common|gpucomp|nscq)|xserver-xorg-video-nvidia|nvidia-driver-pinning)' || true)
  [ -n "$p" ] && sudo dpkg --purge --force-depends --force-remove-reinstreq $p >/dev/null 2>&1 || true
}

unload_modules() {
  sudo systemctl stop nvidia-persistenced nvidia-dcgm 2>/dev/null || true
  # gdrdrv (GPUDirect RDMA) holds the nvidia module on Nebius images.
  sudo rmmod gdrdrv nvidia_peermem nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || true
}

case "$cmd" in
install-run)
  major=${ver%%.*}
  mkdir -p ~/drv && cd ~/drv
  f=NVIDIA-Linux-x86_64-$ver.run
  [ -f "$f" ] || curl -sSfLO "https://us.download.nvidia.com/tesla/$ver/$f" \
    || curl -sSfLO "https://download.nvidia.com/XFree86/Linux-x86_64/$ver/$f"
  sha256sum "$f"
  if command -v nvidia-uninstall >/dev/null; then sudo nvidia-uninstall --silent || true; fi
  purge_apt_driver
  unload_modules
  if [ "$major" -ge 560 ]; then mod="--kernel-module-type=open"; else mod="-m=kernel-open"; fi
  sudo sh "$f" --silent $mod --no-questions --ui=none --no-x-check --no-nouveau-check
  head -1 /proc/driver/nvidia/version
  ;;
install-apt)
  deb=$3
  prefs=/tmp/prefs-$ver
  mkdir -p "$prefs"
  cp /etc/apt/preferences.d/00-kernel-lock "$prefs/"
  purge_apt_driver
  sudo apt-get -o Dir::Etc::PreferencesParts="$prefs" -f install -y -q >/dev/null
  sudo apt-get -o Dir::Etc::PreferencesParts="$prefs" install -y -q --no-install-recommends \
    "nvidia-open=$deb"
  dkms status | grep nvidia || true
  echo "installed; now: sudo reboot"
  ;;
test)
  nvcc=${3:-/usr/local/cuda/bin/nvcc}
  head -1 /proc/driver/nvidia/version
  OG=$HOME/ogkm-$ver
  [ -d "$OG" ] || git clone -q --depth 1 --branch "$ver" \
      https://github.com/NVIDIA/open-gpu-kernel-modules.git "$OG" 2>/dev/null
  cd "$SPIKE"
  rm -f shim/libdynagpu_shim.so workloads/w2_work
  make OGKM="$OG" NVCC="$nvcc -arch=sm_89" shim/libdynagpu_shim.so workloads/w2_work 2>&1 \
    | grep -iE "warn|error" || true
  out=$SPIKE/results/week3/e12-$ver
  mkdir -p "$out"
  echo '{}' > "$out/solo.json"   # the quick checks don't need calibration
  { cat /proc/driver/nvidia/version; nvidia-smi --query-gpu=name,driver_version,vbios_version \
      --format=csv; $nvcc --version | tail -2; } > "$out/environment.txt" 2>&1
  python3 scripts/run_week3.py --spike "$SPIKE" --py python3 --out "$out" --only quick
  sudo dmesg | grep -iE "xid" | tail -5 || true
  ;;
*)
  echo "usage: $0 install-run|install-apt|test <version> ..." >&2; exit 2
  ;;
esac
