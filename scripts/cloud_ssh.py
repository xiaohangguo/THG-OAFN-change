"""Local helper for the SeetaCloud GPU box (password auth via paramiko).

Usage (pwsh):
    $env:CLOUD_HOST='connect.bjb2.seetacloud.com'; $env:CLOUD_PORT='28765';
    $env:CLOUD_USER='root'; $env:CLOUD_PW='...';
    python scripts/cloud_ssh.py run 'nvidia-smi'
    python scripts/cloud_ssh.py get /root/threshold.log .
    python scripts/cloud_ssh.py put local remote

Password is never written to the repo; it lives only in process env.
"""

from __future__ import annotations

import os
import sys
import time

import paramiko


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=os.environ["CLOUD_HOST"],
        port=int(os.environ.get("CLOUD_PORT", "22")),
        username=os.environ["CLOUD_USER"],
        password=os.environ["CLOUD_PW"],
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 600) -> int:
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    chan = stdout.channel
    while True:
        while chan.recv_ready():
            sys.stdout.write(chan.recv(65536).decode("utf-8", "replace"))
            sys.stdout.flush()
        while chan.recv_stderr_ready():
            sys.stderr.write(chan.recv_stderr(65536).decode("utf-8", "replace"))
            sys.stderr.flush()
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        time.sleep(0.2)
    rc = chan.recv_exit_status()
    # drain anything left
    sys.stdout.write(stdout.read().decode("utf-8", "replace"))
    err = stderr.read().decode("utf-8", "replace")
    if err:
        sys.stderr.write(err)
    print(f"\n[exit code: {rc}]")
    return rc


def main() -> int:
    mode = sys.argv[1]
    client = connect()
    try:
        if mode == "run":
            return run(client, sys.argv[2], timeout=int(sys.argv[3]) if len(sys.argv) > 3 else 600)
        if mode in ("get", "put"):
            sftp = client.open_sftp()
            if mode == "get":
                remote, local = sys.argv[2], sys.argv[3]
                sftp.get(remote, local)
            else:
                local, remote = sys.argv[2], sys.argv[3]
                sftp.put(local, remote)
            print(f"OK {mode} {sys.argv[2]} -> {sys.argv[3]}")
            return 0
        print(f"unknown mode {mode}")
        return 2
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
