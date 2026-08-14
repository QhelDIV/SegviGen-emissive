#!/usr/bin/env python3
"""
Cluster SSH helper — supports SFU Solar and Fir (Compute Canada) clusters.
Uses SSH key auth (~/.ssh/id_ed25519). No passwords.

Usage:
  python3 cluster_ssh.py [--cluster solar|fir] run "<cmd>" [<cmd2> ...]
  python3 cluster_ssh.py [--cluster solar|fir] read <remote_path>
  python3 cluster_ssh.py [--cluster solar|fir] write <remote_path>
  python3 cluster_ssh.py [--cluster solar|fir] ls [<remote_path>]
  python3 cluster_ssh.py [--cluster solar|fir] monitor [interval_sec]
  python3 cluster_ssh.py jobs                   # show all jobs, highlight yours
  python3 cluster_ssh.py nodes                  # show node inventory + state
  python3 cluster_ssh.py gpu <node_id>          # GPU utilization on a node (e.g. 09)
  python3 cluster_ssh.py irun <node_id> [gpus]  # interactive session on a node

Fir requires a live ControlMaster socket at ~/.ssh/fir_master.
Establish it once with:
  ssh -M -S ~/.ssh/fir_master -o ControlPersist=yes -o ServerAliveInterval=30 xya120@fir.alliancecan.ca
"""

import sys
import os
import time
import subprocess

# ── Solar config ──────────────────────────────────────────────────────────────
SOLAR_HOST = "solar.cs.sfu.ca"
SOLAR_PORT = 24
SOLAR_USER = "xya120"
SOLAR_KEY  = os.path.expanduser("~/.ssh/id_ed25519")
SOLAR_SOCKET = os.path.expanduser("~/.ssh/solar_master")

# ── Fir config ────────────────────────────────────────────────────────────────
FIR_HOST   = "fir.alliancecan.ca"
FIR_USER   = "xya120"
FIR_SOCKET = os.path.expanduser("~/.ssh/fir_master")


# ══════════════════════════════════════════════════════════════════════════════
# Solar (system ssh via ControlMaster for connection reuse)
# ══════════════════════════════════════════════════════════════════════════════

def _solar_base_cmd() -> list[str]:
    """Base ssh command reusing a ControlMaster socket if alive, else direct key auth."""
    if os.path.exists(SOLAR_SOCKET):
        return ["ssh", "-S", SOLAR_SOCKET, f"{SOLAR_USER}@{SOLAR_HOST}"]
    return [
        "ssh", "-p", str(SOLAR_PORT),
        "-i", SOLAR_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        f"{SOLAR_USER}@{SOLAR_HOST}",
    ]


def solar_ensure_master():
    """Start a ControlMaster socket for Solar if not already running."""
    if os.path.exists(SOLAR_SOCKET):
        result = subprocess.run(
            ["ssh", "-S", SOLAR_SOCKET, "-O", "check", f"{SOLAR_USER}@{SOLAR_HOST}"],
            capture_output=True
        )
        if result.returncode == 0:
            return
    subprocess.Popen([
        "ssh", "-M", "-S", SOLAR_SOCKET,
        "-p", str(SOLAR_PORT),
        "-i", SOLAR_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ControlPersist=600",
        "-o", "ServerAliveInterval=30",
        "-N", f"{SOLAR_USER}@{SOLAR_HOST}",
    ])
    time.sleep(1)


def solar_ssh(cmd: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        _solar_base_cmd() + [cmd],
        capture_output=capture, text=True
    )


def solar_run(commands: list[str]) -> int:
    solar_ensure_master()
    exit_code = 0
    for cmd in commands:
        result = solar_ssh(cmd)
        exit_code = result.returncode
    return exit_code


def solar_read(remote_path: str):
    solar_ensure_master()
    result = solar_ssh(f"cat {remote_path}", capture=True)
    if result.returncode != 0:
        print(f"ERROR reading {remote_path}: {result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
    print(result.stdout, end="")


def solar_write(remote_path: str, content: str):
    solar_ensure_master()
    proc = subprocess.run(
        _solar_base_cmd() + [f"cat > {remote_path}"],
        input=content, text=True, capture_output=True
    )
    if proc.returncode != 0:
        print(f"ERROR writing {remote_path}: {proc.stderr}", file=sys.stderr)
        sys.exit(proc.returncode)


def solar_ls(remote_path: str):
    solar_ensure_master()
    result = solar_ssh(f"ls -la {remote_path}", capture=True)
    if result.returncode != 0:
        print(f"ERROR listing {remote_path}: {result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
    print(result.stdout, end="")


def solar_monitor(interval: int = 30):
    solar_ensure_master()
    print(f"Monitoring Solar jobs for {SOLAR_USER} every {interval}s. Ctrl-C to stop.\n")
    while True:
        result = solar_ssh(
            f"squeue -u {SOLAR_USER} --format='%.18i %.9P %.30j %.8u %.8T %.10M %.6D %R'",
            capture=True
        )
        output = result.stdout.strip()
        print(f"[{time.strftime('%H:%M:%S')}]")
        print(output if output else "  (no jobs running)")
        print()
        if not output or output.count("\n") == 0:
            print("No jobs in queue. Done.")
            break
        time.sleep(interval)


# ══════════════════════════════════════════════════════════════════════════════
# Fir (ControlMaster via system ssh)
# ══════════════════════════════════════════════════════════════════════════════

def fir_check_socket():
    """Raise if the ControlMaster socket is not alive."""
    result = subprocess.run(
        ["ssh", "-S", FIR_SOCKET, "-O", "check", f"{FIR_USER}@{FIR_HOST}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("ERROR: Fir ControlMaster socket not found or dead.", file=sys.stderr)
        print("Run this first (requires phone confirmation):", file=sys.stderr)
        print(f"  ssh -M -S {FIR_SOCKET} -o ControlPersist=yes -o ServerAliveInterval=30 {FIR_USER}@{FIR_HOST}", file=sys.stderr)
        sys.exit(1)


def fir_ssh(cmd: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-S", FIR_SOCKET, f"{FIR_USER}@{FIR_HOST}", cmd],
        capture_output=capture, text=True
    )


def fir_run(commands: list[str]) -> int:
    fir_check_socket()
    exit_code = 0
    for cmd in commands:
        result = fir_ssh(cmd)
        exit_code = result.returncode
    return exit_code


def fir_read(remote_path: str):
    fir_check_socket()
    result = fir_ssh(f"cat {remote_path}", capture=True)
    if result.returncode != 0:
        print(f"ERROR reading {remote_path}: {result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
    print(result.stdout, end="")


def fir_write(remote_path: str, content: str):
    fir_check_socket()
    proc = subprocess.run(
        ["ssh", "-S", FIR_SOCKET, f"{FIR_USER}@{FIR_HOST}", f"cat > {remote_path}"],
        input=content, text=True, capture_output=True
    )
    if proc.returncode != 0:
        print(f"ERROR writing {remote_path}: {proc.stderr}", file=sys.stderr)
        sys.exit(proc.returncode)


def fir_ls(remote_path: str):
    fir_check_socket()
    result = fir_ssh(f"ls -la {remote_path}", capture=True)
    if result.returncode != 0:
        print(f"ERROR listing {remote_path}: {result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
    print(result.stdout, end="")


def fir_monitor(interval: int = 30):
    fir_check_socket()
    print(f"Monitoring Fir jobs for {FIR_USER} every {interval}s. Ctrl-C to stop.\n")
    while True:
        result = fir_ssh(
            f"squeue -u {FIR_USER} --format='%.18i %.9P %.30j %.8u %.8T %.10M %.6D %R'",
            capture=True
        )
        output = result.stdout.strip()
        print(f"[{time.strftime('%H:%M:%S')}]")
        print(output if output else "  (no jobs running)")
        print()
        if not output or output.count("\n") == 0:
            print("No jobs in queue. Done.")
            break
        time.sleep(interval)


# ══════════════════════════════════════════════════════════════════════════════
# Node inventory (from xgutils/misc/solarutils.py)
# ══════════════════════════════════════════════════════════════════════════════

NODES = {
    '01': dict(gpu_type='Quadro RTX 6000', n_gpus=6,  gpu_mem='24GB', partition='cs-gpu-research'),
    '02': dict(gpu_type='RTX 2080Ti',      n_gpus=8,  gpu_mem='11GB', partition='3dlg-hcvc-lab-long'),
    '03': dict(gpu_type='RTX 2080Ti',      n_gpus=4,  gpu_mem='11GB', partition='cs-gpu-research'),
    '05': dict(gpu_type='RTX A5000',       n_gpus=8,  gpu_mem='24GB', partition='3dlg-hcvc-lab-long'),
    '06': dict(gpu_type='RTX A5000',       n_gpus=8,  gpu_mem='24GB', partition='cs-gpu-research'),
    '07': dict(gpu_type='A40',             n_gpus=4,  gpu_mem='48GB', partition='3dlg-hcvc-lab-long'),
    '08': dict(gpu_type='A100',            n_gpus=4,  gpu_mem='80GB', partition='3dlg-hcvc-lab-long'),
    '09': dict(gpu_type='A40',             n_gpus=8,  gpu_mem='48GB', partition='3dlg-hcvc-lab-long'),
    '12': dict(gpu_type='RTX A6000',       n_gpus=2,  gpu_mem='48GB', partition='cs-gpu-research'),
    '13': dict(gpu_type='A40',             n_gpus=4,  gpu_mem='48GB', partition='3dlg-hcvc-lab-long'),
    '14': dict(gpu_type='A40',             n_gpus=4,  gpu_mem='48GB', partition='3dlg-hcvc-lab-long'),
    '15': dict(gpu_type='L40S',            n_gpus=4,  gpu_mem='48GB', partition='3dlg-hcvc-lab-long'),
    '16': dict(gpu_type='L40S',            n_gpus=4,  gpu_mem='48GB', partition='3dlg-hcvc-lab-long'),
    '17': dict(gpu_type='L40S',            n_gpus=4,  gpu_mem='48GB', partition='3dlg-hcvc-lab-long'),
    '18': dict(gpu_type='L40S',            n_gpus=4,  gpu_mem='48GB', partition='3dlg-hcvc-lab-long'),
}


# ══════════════════════════════════════════════════════════════════════════════
# Higher-level commands
# ══════════════════════════════════════════════════════════════════════════════

def solar_jobs():
    """Show all queued/running jobs, highlighting xya120's jobs."""
    solar_ensure_master()
    result = solar_ssh(
        "squeue -o '%.10i %.9P %.32j %.8u %.2t %.10M %.10l %.6D %.10b %.6C %.6m %R' --sort=+N",
        capture=True
    )
    for line in result.stdout.splitlines():
        if SOLAR_USER in line:
            print(f"\033[1;32m{line}\033[0m")
        else:
            print(line)


def solar_nodes():
    """Show node inventory with GPU type and current SLURM state."""
    solar_ensure_master()
    result = solar_ssh(
        "sinfo -p 3dlg-hcvc-lab-long --format='%.12n %.6t %.10e %.10m %G' --noheader",
        capture=True
    )
    state_map = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts:
            node = parts[0].replace('cs-venus-', '')
            state_map[node] = parts[1] if len(parts) > 1 else '?'

    print(f"{'Node':<14} {'GPUs':<5} {'Type':<18} {'VRAM':<8} {'Partition':<22} {'State'}")
    print("-" * 80)
    for nid, info in sorted(NODES.items()):
        state = state_map.get(nid, '-')
        print(f"cs-venus-{nid}   {info['n_gpus']:<5} {info['gpu_type']:<18} {info['gpu_mem']:<8} "
              f"{info['partition']:<22} {state}")


def solar_gpu(node_id: str):
    """Show GPU utilization on a specific node (e.g. '09')."""
    solar_ensure_master()
    node = f"cs-venus-{node_id.zfill(2)}"
    result = solar_ssh(
        f"ssh {node} nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,"
        f"memory.used,memory.total,temperature.gpu --format=csv",
        capture=True
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr)


def solar_irun(node_id: str, n_gpus: int = 1):
    """Start an interactive bash session on a node (like xsrun)."""
    node = f"cs-venus-{node_id.zfill(2)}"
    info = NODES.get(node_id.zfill(2), {})
    partition = info.get('partition', '3dlg-hcvc-lab-long')
    mem = f"{n_gpus * 16}G" if n_gpus > 0 else "32G"
    ntasks = n_gpus if n_gpus > 0 else 1
    gres = f"gpu:{n_gpus}" if n_gpus > 0 else "gpu:0"
    cmd = [
        "ssh", "-p", str(SOLAR_PORT), "-i", SOLAR_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-t", f"{SOLAR_USER}@{SOLAR_HOST}",
        f"srun -J irun --nodelist {node} --gres {gres} --ntasks-per-node {ntasks} "
        f"--cpus-per-task=6 --time=7-00:00:00 --mem={mem} --partition={partition} "
        f"--pty bash -c 'source ~/.bash_aliases && bash'"
    ]
    os.execvp("ssh", cmd)  # replace process so terminal is fully interactive


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = sys.argv[1:]

    # Parse optional --cluster flag
    cluster = "solar"
    if args and args[0] == "--cluster":
        if len(args) < 2:
            print("--cluster requires solar or fir")
            sys.exit(1)
        cluster = args[1]
        args = args[2:]
    elif args and args[0].startswith("--cluster="):
        cluster = args[0].split("=", 1)[1]
        args = args[1:]

    if cluster not in ("solar", "fir"):
        print(f"Unknown cluster: {cluster}. Use solar or fir.")
        sys.exit(1)

    if not args:
        print(__doc__)
        sys.exit(1)

    mode = args[0]
    rest = args[1:]

    if cluster == "solar":
        default_home = f"/home/{SOLAR_USER}"
        run_fn      = solar_run
        read_fn     = solar_read
        write_fn    = solar_write
        ls_fn       = solar_ls
        monitor_fn  = solar_monitor
    else:
        default_home = f"/home/{FIR_USER}"
        run_fn      = fir_run
        read_fn     = fir_read
        write_fn    = fir_write
        ls_fn       = fir_ls
        monitor_fn  = fir_monitor

    if mode == "run":
        if not rest:
            print("Usage: cluster_ssh.py [--cluster solar|fir] run <cmd> [<cmd2> ...]")
            sys.exit(1)
        sys.exit(run_fn(rest))

    elif mode == "read":
        if not rest:
            print("Usage: cluster_ssh.py [--cluster solar|fir] read <remote_path>")
            sys.exit(1)
        read_fn(rest[0])

    elif mode == "write":
        if not rest:
            print("Usage: cluster_ssh.py [--cluster solar|fir] write <remote_path>")
            sys.exit(1)
        content = sys.stdin.read()
        write_fn(rest[0], content)

    elif mode == "ls":
        path = rest[0] if rest else default_home
        ls_fn(path)

    elif mode == "monitor":
        interval = int(rest[0]) if rest else 30
        monitor_fn(interval)

    elif mode == "jobs":
        solar_jobs()

    elif mode == "nodes":
        solar_nodes()

    elif mode == "gpu":
        if not rest:
            print("Usage: cluster_ssh.py gpu <node_id>  (e.g. 09)")
            sys.exit(1)
        solar_gpu(rest[0])

    elif mode == "irun":
        if not rest:
            print("Usage: cluster_ssh.py irun <node_id> [n_gpus]  (e.g. irun 07 1)")
            sys.exit(1)
        n_gpus = int(rest[1]) if len(rest) > 1 else 1
        solar_irun(rest[0], n_gpus)

    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        sys.exit(1)
