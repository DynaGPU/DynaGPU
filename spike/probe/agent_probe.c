// E2 outside routes: can a separate agent process set the timeslice of a
// channel group that belongs to another process?
//
//   Route B: use the app's client + TSG handles directly on the agent's own
//            /dev/nvidiactl fd.
//   Route C: allocate the agent's own RM client and device, duplicate the
//            app's TSG into it (NV_ESC_RM_DUP_OBJECT), then set the timeslice
//            through the duplicate.
//
// Usage: agent_probe <hClient> <hTsg> <timeslice_us> [gpu_minor]
// Prints one line per step with the ioctl return value and RM status.

#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "nvtypes.h"
#include "nvos.h"
#include "nvstatus.h"
#include "nv_escape.h"
#include "nv-ioctl.h"
#include "nv-ioctl-numbers.h"
#include "class/cl0000.h"
#include "class/cl0080.h"
#include "ctrl/ctrla06c.h"

#define MAGIC 'F'

static NvU32 rm_control(int fd, NvHandle c, NvHandle o, NvU32 cmd, void *p, NvU32 sz, int *r) {
  NVOS54_PARAMETERS a = {0};
  a.hClient = c; a.hObject = o; a.cmd = cmd;
  a.params = (NvP64)(uintptr_t)p; a.paramsSize = sz;
  *r = ioctl(fd, _IOWR(MAGIC, NV_ESC_RM_CONTROL, NVOS54_PARAMETERS), &a);
  return a.status;
}

static NvU32 rm_alloc(int fd, NvHandle root, NvHandle parent, NvHandle *obj, NvU32 cls,
                      void *params, NvU32 psz, int *r) {
  NVOS64_PARAMETERS a = {0};
  a.hRoot = root; a.hObjectParent = parent; a.hObjectNew = *obj; a.hClass = cls;
  a.pAllocParms = (NvP64)(uintptr_t)params; a.paramsSize = psz;
  *r = ioctl(fd, _IOWR(MAGIC, NV_ESC_RM_ALLOC, NVOS64_PARAMETERS), &a);
  *obj = a.hObjectNew;
  return a.status;
}

static void try_set_get(const char *route, int fd, NvHandle c, NvHandle t, NvU64 us) {
  int r;
  NVA06C_CTRL_TIMESLICE_PARAMS sp = {0};
  sp.timesliceUs = us;
  NvU32 st = rm_control(fd, c, t, NVA06C_CTRL_CMD_SET_TIMESLICE, &sp, sizeof(sp), &r);
  printf("route=%s step=SET_TIMESLICE us=%llu ioctl=%d status=0x%x %s\n", route,
         (unsigned long long)us, r, st, st == NV_OK ? "OK" : "FAIL");
#ifdef NVA06C_CTRL_CMD_GET_TIMESLICE
  NVA06C_CTRL_TIMESLICE_PARAMS gp = {0};
  st = rm_control(fd, c, t, NVA06C_CTRL_CMD_GET_TIMESLICE, &gp, sizeof(gp), &r);
  printf("route=%s step=GET_TIMESLICE ioctl=%d status=0x%x value_us=%llu\n", route, r, st,
         (unsigned long long)gp.timesliceUs);
#endif
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s hClient hTsg timeslice_us [gpu_minor]\n", argv[0]);
    return 2;
  }
  NvHandle appClient = strtoul(argv[1], NULL, 0), appTsg = strtoul(argv[2], NULL, 0);
  NvU64 us = strtoull(argv[3], NULL, 0);
  int minor = argc > 4 ? atoi(argv[4]) : 0;
  int r;

  int ctl = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
  if (ctl < 0) { perror("open nvidiactl"); return 1; }

  // Route B
  try_set_get("B", ctl, appClient, appTsg, us);

  // Route C: own client
  NvHandle root = 0;
  NvU32 st = rm_alloc(ctl, 0, 0, &root, NV01_ROOT_CLIENT, NULL, 0, &r);
  printf("route=C step=ALLOC_ROOT ioctl=%d status=0x%x hClient=0x%x\n", r, st, root);
  if (st != NV_OK) return 0;

  char path[64];
  snprintf(path, sizeof(path), "/dev/nvidia%d", minor);
  int dev = open(path, O_RDWR | O_CLOEXEC);
  nv_ioctl_register_fd_t reg = {.ctl_fd = ctl};
  r = ioctl(dev, _IOWR(MAGIC, NV_ESC_REGISTER_FD, nv_ioctl_register_fd_t), &reg);
  printf("route=C step=REGISTER_FD dev=%s ioctl=%d\n", path, r);

  NV0080_ALLOC_PARAMETERS dp = {0};
  dp.deviceId = minor;
  dp.hClientShare = root;
  NvHandle device = 0xd0000001;
  st = rm_alloc(ctl, root, root, &device, NV01_DEVICE_0, &dp, sizeof(dp), &r);
  printf("route=C step=ALLOC_DEVICE ioctl=%d status=0x%x hDevice=0x%x\n", r, st, device);
  if (st != NV_OK) return 0;

  NVOS55_PARAMETERS d = {0};
  d.hClient = root;
  d.hParent = device;
  d.hObject = 0xd0000002;
  d.hClientSrc = appClient;
  d.hObjectSrc = appTsg;
  r = ioctl(ctl, _IOWR(MAGIC, NV_ESC_RM_DUP_OBJECT, NVOS55_PARAMETERS), &d);
  printf("route=C step=DUP_TSG ioctl=%d status=0x%x hDup=0x%x\n", r, d.status, d.hObject);
  if (d.status != NV_OK) return 0;

  try_set_get("C", ctl, root, d.hObject, us);
  return 0;
}
