# /cluster — Operate on SFU Solar and Fir HPC clusters

Use this skill to manage jobs on SLURM clusters.
Helper script: `~/.claude/cluster_ssh.py`

## Clusters

| | Solar | Fir |
|---|---|---|
| GPU | L40S (44GB) | H100 |
| Account flag | (none) | `--account=rrg-msavva` |
| Repo path | `/home/sya225/on-policy-distillation` | `/home/sya225/scratch/on-policy-distillation` |
| Env | `source ~/.bashrc` + `source test_env/bin/activate` | `module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv` + `source test/bin/activate` |
| Connection | paramiko (automatic) | SSH ControlMaster socket |

Add `--cluster fir` to any command to target Fir. Default is Solar.

```bash
python3 ~/.claude/cluster_ssh.py [--cluster solar|fir] <mode> ...
```

---

## Fir: One-time login setup (requires phone 2FA)

Fir requires Duo/phone confirmation on every new SSH session. Use ControlMaster to authenticate once and reuse the connection:

```bash
# In a local tmux session (keeps socket alive):
tmux new -s fir
ssh -M -S ~/.ssh/fir_master -o ControlPersist=yes -o ServerAliveInterval=30 sya225@fir.alliancecan.ca
# Confirm on phone → Ctrl+B D to detach

# Check socket is alive:
ssh -S ~/.ssh/fir_master -O check sya225@fir.alliancecan.ca
```

After this, all `--cluster fir` commands work without any authentication.

---

## Available operations

### Run commands
```bash
python3 ~/.claude/cluster_ssh.py run "<cmd>"
python3 ~/.claude/cluster_ssh.py run "<cmd1>" "<cmd2>"   # chained, state persists (Solar only)

python3 ~/.claude/cluster_ssh.py --cluster fir run "<cmd>"
```

### Read/write remote files
```bash
python3 ~/.claude/cluster_ssh.py read /path/on/cluster/file.sh
echo "content" | python3 ~/.claude/cluster_ssh.py write /path/on/cluster/file.sh
cat local_file.sh | python3 ~/.claude/cluster_ssh.py write /path/on/cluster/file.sh

# Fir:
python3 ~/.claude/cluster_ssh.py --cluster fir read /path/on/fir/file.sh
cat local_file.sh | python3 ~/.claude/cluster_ssh.py --cluster fir write /path/on/fir/file.sh
```

### List remote directory
```bash
python3 ~/.claude/cluster_ssh.py ls /path/on/cluster/
python3 ~/.claude/cluster_ssh.py --cluster fir ls /path/on/fir/
```

### Monitor jobs until completion
```bash
python3 ~/.claude/cluster_ssh.py monitor [interval_seconds]
python3 ~/.claude/cluster_ssh.py --cluster fir monitor [interval_seconds]
```

---

## Solar: Login node vs compute node (SEPARATE filesystems)

`/home/sya225` on the login node and on cs-venus-15 are **different filesystems**. Files written
via SFTP (`cluster_ssh.py write`) only go to the login node. To read/write files that jobs use,
you MUST use `srun -w <node>`.

### Non-exclusive job submission (allow parallel srun access)

When submitting eval or short jobs, do **NOT** use `--exclusive`. Use explicit resource requests:

```bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Do NOT add: #SBATCH --exclusive
```

This allows `srun` access to the same node while the job is running. With `--exclusive`, any
`srun -w <node>` blocks until the job finishes.

### Run a command ON a specific Solar compute node (e.g. cs-venus-15)

> **DO NOT** use `salloc ... bash -c '...'` — that runs on the login node.
> **DO** use `srun --ntasks=1 -w <node> bash -c '...'`.

```bash
# Run a single command on cs-venus-15
python3 ~/.claude/cluster_ssh.py run "srun --ntasks=1 --gres=gpu:l40s:4 --cpus-per-task=16 --mem=192G --exclusive -w cs-venus-15 bash -c '<cmd>'"

# Read a log file from cs-venus-15
python3 ~/.claude/cluster_ssh.py run "srun --ntasks=1 --gres=gpu:l40s:4 --cpus-per-task=16 --mem=192G --exclusive -w cs-venus-15 bash -c 'tail -100 /home/sya225/on-policy-distillation/logs/cs-venus-15-<JOBID>.out'"

# Check output directory / checkpoints
python3 ~/.claude/cluster_ssh.py run "srun --ntasks=1 --gres=gpu:l40s:4 --cpus-per-task=16 --mem=192G --exclusive -w cs-venus-15 bash -c 'find /home/sya225/on-policy-distillation/output -name checkpoint-* -type d'"
```

**Why `--ntasks=1`**: without it, srun spawns one task per GPU (4 tasks for 4 GPUs), printing output 4 times.

### Write a file directly to cs-venus-15 (base64)

SFTP writes only reach the login node. Use base64 to write to compute node:

```bash
CONTENT=$(base64 -w0 << 'EOF'
file content here
EOF
)
python3 ~/.claude/cluster_ssh.py run "srun --ntasks=1 --gres=gpu:l40s:4 --cpus-per-task=16 --mem=192G --exclusive -w cs-venus-15 bash -c 'echo $CONTENT | base64 -d > /home/sya225/path/to/file.sh && echo done'"
```

---

## Typical workflow

### Solar
1. **Read** the existing sbatch file from the login node
2. **Edit** it: `write` mode pushes back to login node
3. **Submit** with `run "sbatch /path/to/job.sh"`
4. **Monitor** with `run "squeue -u sya225"` or `monitor`
5. **Read logs/results** from compute node with `srun --ntasks=1 -w <node> bash -c '...'`

### Fir
1. Ensure ControlMaster socket is alive (`ssh -O check`)
2. Same workflow with `--cluster fir` prefix
3. Fir filesystems ARE shared (NFS) — no need for srun to read files

$ARGUMENTS
