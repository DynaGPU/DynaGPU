// DynaGPU spike shim (LD_PRELOAD).
//
// E1: watch the app's ioctl calls on /dev/nvidiactl and record every channel
//     group (TSG, class KEPLER_CHANNEL_GROUP_A = 0xa06c) the driver creates.
// E2: set the channel group's timeslice from inside the app ("in-process
//     route"), using the app's own RM client and file descriptor.
//
// It never touches kernel launches: it only looks at the one-time RM_ALLOC
// calls, so there is no per-launch cost.
//
// Environment:
//   DYNAGPU_LOG=<file>          log file (default: stderr)
//   DYNAGPU_TIMESLICE_US=<n>    set this timeslice on every new channel group
//   DYNAGPU_CTL=<file>          poll this file; when it holds a new number,
//                               apply it as the timeslice (runtime changes)
//   DYNAGPU_STATE=<file>        append "pid fd hClient hTsg engineType" lines
//                               so an outside agent can try routes B and C
//
// Built against NVIDIA's open-gpu-kernel-modules headers (MIT).

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include "nvtypes.h"
#include "nvos.h"
#include "nvstatus.h"
#include "nv_escape.h"
#include "class/cla06c.h"
#include "alloc/alloc_channel.h"
#include "class/cl2080_notification.h"
#include "ctrl/ctrla06c.h"

#define NV_IOCTL_MAGIC 'F'
#define MAX_TSG 64

typedef int (*ioctl_fn)(int, unsigned long, ...);
static ioctl_fn real_ioctl;

struct tsg {
  int fd;
  NvHandle hClient, hTsg;
  NvU32 engineType;
};
static struct tsg tsgs[MAX_TSG];
static int ntsg;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static FILE *logf;
static pthread_once_t once = PTHREAD_ONCE_INIT;

static void logmsg(const char *fmt, ...) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  FILE *f = logf ? logf : stderr;
  fprintf(f, "[dynagpu %d %ld.%06ld] ", getpid(), (long)ts.tv_sec, ts.tv_nsec / 1000);
  va_list ap;
  va_start(ap, fmt);
  vfprintf(f, fmt, ap);
  va_end(ap);
  fputc('\n', f);
  fflush(f);
}

// Send an RM control call on the app's own fd and client.
static NvU32 rm_control(int fd, NvHandle hClient, NvHandle hObject, NvU32 cmd,
                        void *params, NvU32 size, int *ioctl_ret) {
  NVOS54_PARAMETERS p;
  memset(&p, 0, sizeof(p));
  p.hClient = hClient;
  p.hObject = hObject;
  p.cmd = cmd;
  p.params = (NvP64)(uintptr_t)params;
  p.paramsSize = size;
  int r = real_ioctl(fd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_CONTROL, NVOS54_PARAMETERS), &p);
  if (ioctl_ret) *ioctl_ret = r;
  return p.status;
}

static void set_timeslice(struct tsg *t, NvU64 us) {
  NVA06C_CTRL_TIMESLICE_PARAMS sp = {0};
  sp.timesliceUs = us;
  int r;
  NvU32 st = rm_control(t->fd, t->hClient, t->hTsg, NVA06C_CTRL_CMD_SET_TIMESLICE,
                        &sp, sizeof(sp), &r);
  logmsg("SET_TIMESLICE tsg=0x%x engine=%u us=%llu ioctl=%d status=0x%x%s", t->hTsg,
         t->engineType, (unsigned long long)us, r, st, st == NV_OK ? " OK" : " FAIL");
#ifdef NVA06C_CTRL_CMD_GET_TIMESLICE
  NVA06C_CTRL_TIMESLICE_PARAMS gp = {0};
  st = rm_control(t->fd, t->hClient, t->hTsg, NVA06C_CTRL_CMD_GET_TIMESLICE, &gp,
                  sizeof(gp), &r);
  logmsg("GET_TIMESLICE tsg=0x%x ioctl=%d status=0x%x value_us=%llu", t->hTsg, r, st,
         (unsigned long long)gp.timesliceUs);
#endif
}

static void *ctl_thread(void *arg) {
  const char *path = (const char *)arg;
  long long last = -1;
  for (;;) {
    FILE *f = fopen(path, "r");
    if (f) {
      long long v;
      if (fscanf(f, "%lld", &v) == 1 && v > 0 && v != last) {
        last = v;
        pthread_mutex_lock(&mu);
        for (int i = 0; i < ntsg; i++)
          if (tsgs[i].engineType == NV2080_ENGINE_TYPE_GRAPHICS)
            set_timeslice(&tsgs[i], (NvU64)v);
        pthread_mutex_unlock(&mu);
      }
      fclose(f);
    }
    usleep(50 * 1000);
  }
  return NULL;
}

static void init(void) {
  real_ioctl = (ioctl_fn)dlsym(RTLD_NEXT, "ioctl");
  const char *lp = getenv("DYNAGPU_LOG");
  if (lp) logf = fopen(lp, "a");
  logmsg("shim loaded");
  const char *ctl = getenv("DYNAGPU_CTL");
  if (ctl) {
    pthread_t th;
    pthread_create(&th, NULL, ctl_thread, (void *)ctl);
    pthread_detach(th);
  }
}

int ioctl(int fd, unsigned long request, ...) {
  va_list ap;
  va_start(ap, request);
  void *arg = va_arg(ap, void *);
  va_end(ap);
  pthread_once(&once, init);

  // CUDA sets its own timeslice on the compute channel group right after
  // creating it. If we have a target, replace CUDA's value with ours before
  // the call goes to the driver, so CUDA cannot overwrite our setting.
  if (_IOC_TYPE(request) == NV_IOCTL_MAGIC && arg && _IOC_NR(request) == NV_ESC_RM_CONTROL &&
      _IOC_SIZE(request) == sizeof(NVOS54_PARAMETERS)) {
    NVOS54_PARAMETERS *c = (NVOS54_PARAMETERS *)arg;
    if (c->cmd == NVA06C_CTRL_CMD_SET_TIMESLICE && c->params) {
      NVA06C_CTRL_TIMESLICE_PARAMS *tp = (NVA06C_CTRL_TIMESLICE_PARAMS *)(uintptr_t)c->params;
      const char *ts = getenv("DYNAGPU_TIMESLICE_US");
      NvU64 want = ts ? (NvU64)strtoull(ts, NULL, 10) : 0;
      logmsg("APP_SET_TIMESLICE hObject=0x%x app_value_us=%llu%s", c->hObject,
             (unsigned long long)tp->timesliceUs, want ? " (overridden)" : "");
      if (want) tp->timesliceUs = want;
    }
  }

  int ret = real_ioctl(fd, request, arg);

  if (_IOC_TYPE(request) != NV_IOCTL_MAGIC || arg == NULL) return ret;
  unsigned nr = _IOC_NR(request);
  unsigned size = _IOC_SIZE(request);

  if (nr == NV_ESC_RM_ALLOC && ret == 0) {
    // NVOS21 and NVOS64 share the leading fields.
    NVOS21_PARAMETERS *p = (NVOS21_PARAMETERS *)arg;
    NvU32 status = (size == sizeof(NVOS64_PARAMETERS)) ? ((NVOS64_PARAMETERS *)arg)->status
                                                       : p->status;
    if (p->hClass == KEPLER_CHANNEL_GROUP_A && status == NV_OK) {
      NvU32 engine = 0xffffffff;
      if (p->pAllocParms) {
        NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS *ap2 =
            (NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS *)(uintptr_t)p->pAllocParms;
        engine = ap2->engineType;
      }
      pthread_mutex_lock(&mu);
      struct tsg *t = NULL;
      if (ntsg < MAX_TSG) {
        t = &tsgs[ntsg++];
        t->fd = fd;
        t->hClient = p->hRoot;
        t->hTsg = p->hObjectNew;
        t->engineType = engine;
      }
      pthread_mutex_unlock(&mu);
      logmsg("TSG_ALLOC fd=%d hClient=0x%x hParent=0x%x hTsg=0x%x engineType=%u", fd,
             p->hRoot, p->hObjectParent, p->hObjectNew, engine);
      const char *sp = getenv("DYNAGPU_STATE");
      if (sp && t) {
        FILE *f = fopen(sp, "a");
        if (f) {
          fprintf(f, "%d %d 0x%x 0x%x %u\n", getpid(), fd, t->hClient, t->hTsg, engine);
          fclose(f);
        }
      }
      const char *ts = getenv("DYNAGPU_TIMESLICE_US");
      // Only the compute/graphics channel group gets a timeslice; copy
      // engines run on their own runlists.
      if (ts && t && engine == NV2080_ENGINE_TYPE_GRAPHICS)
        set_timeslice(t, (NvU64)strtoull(ts, NULL, 10));
    }
  } else if (nr == NV_ESC_RM_CONTROL && size == sizeof(NVOS54_PARAMETERS)) {
    // Log any channel-group controls the CUDA driver sends itself.
    NVOS54_PARAMETERS *p = (NVOS54_PARAMETERS *)arg;
    if ((p->cmd >> 16) == 0xa06c)
      logmsg("APP_CONTROL cmd=0x%x hObject=0x%x status=0x%x", p->cmd, p->hObject, p->status);
  }
  return ret;
}
