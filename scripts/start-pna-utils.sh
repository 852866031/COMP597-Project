SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

# Pin the job to 1 CPU so that per-process CPU util (psutil.Process.cpu_percent)
# reflects the utilisation of exactly the one core allocated to this task.
# We achieve this by writing a minimal override config and pointing
# COMP597_SLURM_CONFIG at it; srun.sh sources it after the defaults, so the
# value wins.  The temp file is cleaned up on exit.
_UTILS_CPU_CFG=$(mktemp)
echo 'export COMP597_SLURM_CPUS_PER_TASK=1' > "${_UTILS_CPU_CFG}"
export COMP597_SLURM_CONFIG="${_UTILS_CPU_CFG}"
trap 'rm -f "${_UTILS_CPU_CFG}"' EXIT

### run PNA Trainer with hardware utilisation stats (pna_utils)
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model pna \
    --trainer pna \
    --data pna_dataset \
    --model_configs.pna.epochs 5 \
    --model_configs.pna.batch_size 256 \
    --trainer_stats pna_utils \
    --trainer_stats_configs.codecarbon.run_num 1 \
    --trainer_stats_configs.codecarbon.project_name PNA \
    --trainer_stats_configs.codecarbon.output_dir '${REPO_DIR}/pna_result'
