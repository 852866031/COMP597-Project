SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

# Parse -bs <batch_size> (default: 4096)
BS=4096
while [[ $# -gt 0 ]]; do
    case "$1" in
        -bs) BS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "${BS}" == "all" ]]; then
    for _bs in 512 1024 2048 4096; do
        echo "==> $(basename "${BASH_SOURCE[0]}") -bs ${_bs}"
        bash "${BASH_SOURCE[0]}" -bs "${_bs}"
    done
    exit $?
fi

# Scale epoch count with batch size (reference: 15 epochs at bs=512).
# Larger batch → more epochs; smaller batch → fewer. Floor at 1, cap at 40.
# Formula: round(15 * BS / 512) using integer arithmetic.
EPOCHS=$(( (15 * BS + 256) / 512 ))
if [[ ${EPOCHS} -lt 1  ]]; then EPOCHS=1;  fi
if [[ ${EPOCHS} -gt 30 ]]; then EPOCHS=30; fi

### run PNA Carbon Trainer (energy consumption measurement)
EPOCHS=120
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model pna \
    --trainer pna \
    --data pna_dataset \
    --model_configs.pna.epochs ${EPOCHS} \
    --model_configs.pna.batch_size ${BS} \
    --model_configs.pna.num_workers 2 \
    --trainer_stats pna_carbon \
    --trainer_stats_configs.codecarbon.run_num 1 \
    --trainer_stats_configs.codecarbon.project_name PNA \
    --trainer_stats_configs.codecarbon.output_dir '${REPO_DIR}/pna_result'
