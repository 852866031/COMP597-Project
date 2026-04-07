SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

# Parse -bs <batch_size> (default: 512)
BS=512
while [[ $# -gt 0 ]]; do
    case "$1" in
        -bs) BS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Scale epoch count with batch size (reference: 5 epochs at bs=256).
# Larger batch → more epochs; smaller batch → fewer. Floor at 1.
# Formula: round(5 * BS / 256) using integer arithmetic.
EPOCHS=$(( (5 * BS + 128) / 256 ))
if [[ ${EPOCHS} -lt 1 ]]; then EPOCHS=1; fi

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
    --model_configs.pna.epochs ${EPOCHS} \
    --model_configs.pna.batch_size ${BS} \
    --trainer_stats pna_utils \
    --trainer_stats_configs.codecarbon.run_num 1 \
    --trainer_stats_configs.codecarbon.project_name PNA \
    --trainer_stats_configs.codecarbon.output_dir '${REPO_DIR}/pna_result'
