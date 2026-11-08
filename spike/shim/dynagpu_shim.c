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
//   DYNAGPU_SOCK=<path>         listen on this Unix datagram socket for
//                               commands from the agent (week 2):
//                                 pause      disable scheduling + preempt now
//                                 pause-nopreempt  disable scheduling only
//                                 resume     enable scheduling again
//                                 preempt    preempt once, stay scheduled
//                                 ts <us>    set timeslice
//                                 il <0|1|2> set interleave level (LOW/MED/HIGH)
//                               each command gets a reply:
//                                 "ok status=0x.. us=<time spent in driver>"
//                                 hb         heartbeat: only renews the lease
//                               "%p" in the path is replaced by the pid
//   DYNAGPU_INTERLEAVE=<0|1|2>  interleave level for new compute groups
//   DYNAGPU_LEASE_MS=<n>        fail open (week 3, E10): once an agent has
//                               sent a command, every command renews a lease
//                               of n ms. If it runs out while the app is
//                               paused or has a changed timeslice, the shim
//                               resumes the app and restores CUDA's own
//                               timeslice by itself.
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
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "nvtypes.h"
#include "nvos.h"
#include "nvstatus.h"
#include "nv_escape.h"
#include "class/cla06c.h"
#include "alloc/alloc_channel.h"
#include "class/cl2080_notification.h"
#include "class/cl2080.h"
#include "ctrl/ctrl2080/ctrl2080fifo.h"
#include "ctrl/ctrla06c.h"

#define NV_IOCTL_MAGIC 'F'
#define MAX_TSG 64

typedef int (*ioctl_fn)(int, unsigned long, ...);
static ioctl_fn real_ioctl;

#define MAX_CHAN 16

struct tsg {
  int fd;
  NvHandle hClient, hTsg;
  NvHandle hDevice;  // parent device: picks the subdevice of the same GPU
  NvU32 engineType;
  NvHandle hChannels[MAX_CHAN];  // GPFIFO channels allocated inside this group
  int nchan;
};
static struct tsg tsgs[MAX_TSG];
static int ntsg;
// The app's subdevices (class NV20_SUBDEVICE_0), needed for DISABLE_CHANNELS.
// CUDA may create a short-lived client first, and on a multi-GPU machine it
// creates one device + subdevice per GPU under the same client. So keep them
// all and use the subdevice with the same client AND parent device as the
// channel group.
#define MAX_SUBDEV 32
static struct { int fd; NvHandle hClient, hDevice, hSubdev; } subdevs[MAX_SUBDEV];
static int nsubdev;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static FILE *logf;
static pthread_once_t once = PTHREAD_ONCE_INIT;

// Lease state (E10). Guarded by mu.
static long lease_ms;          // 0 = no lease
static double lease_last_us;   // last command from the agent (monotonic)
static int lease_armed;        // an agent has talked to us
static int st_paused;          // we paused the app
static int st_ts_changed;      // the agent changed the timeslice
static NvU64 app_default_ts = 2048;  // CUDA's own value, seen in its SET_TIMESLICE
static char sock_path[108];

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

static double mono_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

// Apply one control to every compute channel group; return the last status.
static NvU32 for_compute(NvU32 cmd, void *params, NvU32 size) {
  NvU32 st = NV_OK;
  int r;
  for (int i = 0; i < ntsg; i++) {
    if (tsgs[i].engineType != NV2080_ENGINE_TYPE_GRAPHICS) continue;
    NvU32 s = rm_control(tsgs[i].fd, tsgs[i].hClient, tsgs[i].hTsg, cmd, params, size, &r);
    if (s != NV_OK || r != 0) st = s ? s : 0xffffffff;
  }
  return st;
}

static NvU32 do_schedule(NvBool enable) {
  NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS p = {0};
  p.bEnable = enable;
  return for_compute(NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, &p, sizeof(p));
}

// Pause/resume through the subdevice: disable or enable every channel of the
// app's compute channel groups. only_sched=false also preempts them off the
// GPU before returning.
static NvU32 do_disable_channels(NvBool disable, NvBool only_sched) {
  NvU32 st = 0xfffffffc;  // no compute channels seen yet
  for (int i = 0; i < ntsg; i++) {
    if (tsgs[i].engineType != NV2080_ENGINE_TYPE_GRAPHICS || tsgs[i].nchan == 0) continue;
    int sd = -1;
    for (int k = nsubdev - 1; k >= 0; k--)
      if (subdevs[k].hClient == tsgs[i].hClient && subdevs[k].hDevice == tsgs[i].hDevice) {
        sd = k;
        break;
      }
    if (sd < 0) return 0xfffffffd;  // no subdevice under this client
    NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS p;
    memset(&p, 0, sizeof(p));
    p.bDisable = disable;
    p.bOnlyDisableScheduling = only_sched;
    for (int c = 0; c < tsgs[i].nchan && c < NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES; c++) {
      p.hClientList[p.numChannels] = tsgs[i].hClient;
      p.hChannelList[p.numChannels] = tsgs[i].hChannels[c];
      p.numChannels++;
    }
    int r;
    st = rm_control(subdevs[sd].fd, subdevs[sd].hClient, subdevs[sd].hSubdev,
                    NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS, &p, sizeof(p), &r);
    if (r != 0) st = 0xffffffff;
    if (st != NV_OK) return st;
  }
  return st;
}

static NvU32 do_preempt(void) {
  NVA06C_CTRL_PREEMPT_PARAMS p = {0};
  p.bWait = NV_TRUE;
  return for_compute(NVA06C_CTRL_CMD_PREEMPT, &p, sizeof(p));
}

static NvU32 do_interleave(NvU32 level) {
  NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS p = {0};
  p.tsgInterleaveLevel = level;
  return for_compute(NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL, &p, sizeof(p));
}

static void *sock_thread(void *arg) {
  const char *path = (const char *)arg;
  int s = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
  unlink(path);
  if (bind(s, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
    logmsg("socket bind %s failed: %s", path, strerror(errno));
    return NULL;
  }
  logmsg("listening on %s", path);
  for (;;) {
    char buf[64] = {0};
    struct sockaddr_un from;
    socklen_t flen = sizeof(from);
    ssize_t n = recvfrom(s, buf, sizeof(buf) - 1, 0, (struct sockaddr *)&from, &flen);
    if (n <= 0) continue;
    double t0 = mono_us();
    NvU32 st = 0xfffffffe;  // unknown command
    pthread_mutex_lock(&mu);
    lease_last_us = mono_us();
    lease_armed = 1;
    if (!strncmp(buf, "hb", 2)) {
      st = NV_OK;
    } else if (!strncmp(buf, "pause-tsg", 9)) {
      // Week-2 first attempt: channel-group schedule off + preempt.
      st = do_schedule(NV_FALSE);
      if (st == NV_OK) st = do_preempt();
    } else if (!strncmp(buf, "pause-sched", 11)) {
      st = do_disable_channels(NV_TRUE, NV_TRUE);   // stop scheduling, don't preempt
    } else if (!strncmp(buf, "pause", 5)) {
      st = do_disable_channels(NV_TRUE, NV_FALSE);  // stop scheduling + preempt now
      if (st == NV_OK) st_paused = 1;
    } else if (!strncmp(buf, "resume-tsg", 10)) {
      st = do_schedule(NV_TRUE);
    } else if (!strncmp(buf, "resume", 6)) {
      st = do_disable_channels(NV_FALSE, NV_FALSE);
      if (st == NV_OK) st_paused = 0;
    } else if (!strncmp(buf, "preempt", 7)) {
      st = do_preempt();
    } else if (!strncmp(buf, "ts ", 3)) {
      NVA06C_CTRL_TIMESLICE_PARAMS p = {0};
      p.timesliceUs = strtoull(buf + 3, NULL, 10);
      st = for_compute(NVA06C_CTRL_CMD_SET_TIMESLICE, &p, sizeof(p));
      st_ts_changed = 1;
    } else if (!strncmp(buf, "il ", 3)) {
      st = do_interleave((NvU32)strtoul(buf + 3, NULL, 10));
    }
    int have = 0;
    for (int i = 0; i < ntsg; i++) have |= tsgs[i].engineType == NV2080_ENGINE_TYPE_GRAPHICS;
    pthread_mutex_unlock(&mu);
    double dt = mono_us() - t0;
    char rep[96];
    int len = snprintf(rep, sizeof(rep), "%s status=0x%x us=%.1f groups=%d",
                       st == NV_OK && have ? "ok" : "err", st, dt, have);
    if (flen > sizeof(sa_family_t))
      sendto(s, rep, len, 0, (struct sockaddr *)&from, flen);
  }
  return NULL;
}

// Fail open: if the agent stops talking, undo what it did.
static void *lease_thread(void *arg) {
  (void)arg;
  for (;;) {
    usleep(50 * 1000);
    pthread_mutex_lock(&mu);
    double now = mono_us();
    if (lease_armed && (st_paused || st_ts_changed) &&
        now - lease_last_us > lease_ms * 1000.0) {
      double age = (now - lease_last_us) / 1000.0;
      NvU32 rs = NV_OK, ts = NV_OK;
      if (st_paused) rs = do_disable_channels(NV_FALSE, NV_FALSE);
      if (st_ts_changed) {
        NVA06C_CTRL_TIMESLICE_PARAMS p = {0};
        p.timesliceUs = app_default_ts;
        ts = for_compute(NVA06C_CTRL_CMD_SET_TIMESLICE, &p, sizeof(p));
      }
      logmsg("LEASE_EXPIRED age_ms=%.1f was_paused=%d resume_status=0x%x "
             "ts_reset_us=%llu ts_status=0x%x took_us=%.1f",
             age, st_paused, rs, (unsigned long long)(st_ts_changed ? app_default_ts : 0), ts,
             mono_us() - now);
      st_paused = 0;
      st_ts_changed = 0;
      lease_armed = 0;  // until the agent talks again
    }
    pthread_mutex_unlock(&mu);
  }
  return NULL;
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
  const char *sock = getenv("DYNAGPU_SOCK");
  if (sock) {
    // Replace "%p" with the pid, so several processes started with the same
    // environment (e.g. torchrun ranks) get their own sockets.
    size_t o = 0;
    for (const char *c = sock; *c && o + 12 < sizeof(sock_path); c++) {
      if (c[0] == '%' && c[1] == 'p') {
        o += snprintf(sock_path + o, sizeof(sock_path) - o, "%d", getpid());
        c++;
      } else {
        sock_path[o++] = *c;
      }
    }
    sock_path[o] = 0;
    pthread_t th;
    pthread_create(&th, NULL, sock_thread, (void *)sock_path);
    pthread_detach(th);
  }
  const char *lease = getenv("DYNAGPU_LEASE_MS");
  if (lease && atol(lease) > 0) {
    lease_ms = atol(lease);
    logmsg("lease %ld ms", lease_ms);
    pthread_t th;
    pthread_create(&th, NULL, lease_thread, NULL);
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
      if (tp->timesliceUs) app_default_ts = tp->timesliceUs;  // remember CUDA's value
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
    if (p->hClass == NV20_SUBDEVICE_0 && status == NV_OK) {
      pthread_mutex_lock(&mu);
      if (nsubdev < MAX_SUBDEV) {
        subdevs[nsubdev].fd = fd;
        subdevs[nsubdev].hClient = p->hRoot;
        subdevs[nsubdev].hDevice = p->hObjectParent;
        subdevs[nsubdev].hSubdev = p->hObjectNew;
        nsubdev++;
      }
      pthread_mutex_unlock(&mu);
      logmsg("SUBDEVICE_ALLOC fd=%d hClient=0x%x hDevice=0x%x hSubdev=0x%x", fd, p->hRoot,
             p->hObjectParent, p->hObjectNew);
    } else if ((p->hClass & 0xff) == 0x6f && status == NV_OK) {
      // A GPFIFO channel (classes xx6f); remember it if it lives in a known group.
      pthread_mutex_lock(&mu);
      for (int i = 0; i < ntsg; i++) {
        if (tsgs[i].hClient == p->hRoot && tsgs[i].hTsg == p->hObjectParent &&
            tsgs[i].nchan < MAX_CHAN) {
          tsgs[i].hChannels[tsgs[i].nchan++] = p->hObjectNew;
          logmsg("CHANNEL_ALLOC class=0x%x hTsg=0x%x hChannel=0x%x engineType=%u", p->hClass,
                 p->hObjectParent, p->hObjectNew, tsgs[i].engineType);
        }
      }
      pthread_mutex_unlock(&mu);
    }
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
        t->hDevice = p->hObjectParent;
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
      const char *il = getenv("DYNAGPU_INTERLEAVE");
      if (il && t && engine == NV2080_ENGINE_TYPE_GRAPHICS) {
        NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS ip = {0};
        ip.tsgInterleaveLevel = (NvU32)strtoul(il, NULL, 10);
        int r;
        NvU32 st = rm_control(t->fd, t->hClient, t->hTsg, NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL,
                              &ip, sizeof(ip), &r);
        logmsg("SET_INTERLEAVE tsg=0x%x level=%u ioctl=%d status=0x%x", t->hTsg,
               ip.tsgInterleaveLevel, r, st);
      }
    }
  } else if (nr == NV_ESC_RM_CONTROL && size == sizeof(NVOS54_PARAMETERS)) {
    // Log any channel-group controls the CUDA driver sends itself.
    NVOS54_PARAMETERS *p = (NVOS54_PARAMETERS *)arg;
    if ((p->cmd >> 16) == 0xa06c)
      logmsg("APP_CONTROL cmd=0x%x hObject=0x%x status=0x%x", p->cmd, p->hObject, p->status);
  }
  return ret;
}
