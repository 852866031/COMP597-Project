#!/bin/bash
# node-info.sh — Submit a job to gpu-teach-03 that prints GPU model, CPU, and memory info.

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

DEFAULT_CONFIG_FILE=${REPO_DIR}/config/default_srun_config.sh

. ${DEFAULT_CONFIG_FILE}

if [[ -f ${COMP597_SLURM_CONFIG} ]]; then
    . ${COMP597_SLURM_CONFIG}
fi

module load slurm

srun \
    --partition=${COMP597_SLURM_PARTITION} \
    --mem=${COMP597_SLURM_MIN_MEM} \
    --time=00:05 \
    --ntasks=1 \
    --account=${COMP597_SLURM_ACCOUNT} \
    --nodelist=${COMP597_SLURM_NODELIST} \
    --cpus-per-task=1 \
    --qos=${COMP597_SLURM_QOS} \
    --gpus=1 \
    bash -c '
echo "============================================"
echo "  Node: $(hostname)"
echo "============================================"
echo
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
echo
echo "--- CPU ---"
lscpu | grep -E "^(Model name|Architecture|CPU\(s\)|Thread|Core|Socket|CPU MHz|CPU max)"
echo
echo "--- Memory ---"
free -h | head -2
echo
echo "--- OS ---"
uname -srm
'
