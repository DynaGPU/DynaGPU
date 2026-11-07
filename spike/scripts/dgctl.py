#!/usr/bin/env python3
"""Tiny client for the shim's command socket: dgctl.py <sock> <command...>"""
import os, socket, sys
me = f"/tmp/dgctl.{os.getpid()}.sock"
s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM); s.bind(me); s.settimeout(1.0)
try:
    s.sendto(" ".join(sys.argv[2:]).encode(), sys.argv[1]); print(s.recv(128).decode())
finally:
    os.unlink(me)
