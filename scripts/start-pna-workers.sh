SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

# Parse -wk <num_workers> (default: 2)
WK=2
while [[ $# -gt 0 ]]; do
    case "$1" in
        -wk) WK="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "${WK}" == "all" ]]; then
    for _wk in 0 2 4; do
        echo "==> $(basename "${BASH_SOURCE[0]}") -wk ${_wk}"
        bash "${BASH_SOURCE[0]}" -wk "${_wk}"
    done
    exit $?
fi

# Fixed batch size for worker sweep; epochs scaled to bs=4096 reference (15 epochs).
BS=4096
EPOCHS=$(( (15 * BS + 256) / 512 ))
if [[ ${EPOCHS} -lt 1  ]]; then EPOCHS=1;  fi
if [[ ${EPOCHS} -gt 30 ]]; then EPOCHS=30; fi

echo "==> utils  bs=${BS} wk=${WK} epochs=${EPOCHS}"
### run PNA Trainer with hardware utilisation stats (pna_utils)
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model pna \
    --trainer pna \
    --data pna_dataset \
    --model_configs.pna.epochs ${EPOCHS} \
    --model_configs.pna.batch_size ${BS} \
    --model_configs.pna.num_workers ${WK} \
    --trainer_stats pna_utils \
    --trainer_stats_configs.codecarbon.run_num 1 \
    --trainer_stats_configs.codecarbon.project_name PNA \
    --trainer_stats_configs.codecarbon.output_dir '${REPO_DIR}/pna_result'

echo "==> carbon bs=${BS} wk=${WK} epochs=${EPOCHS}"
### run PNA Carbon Trainer (energy consumption measurement)
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model pna \
    --trainer pna \
    --data pna_dataset \
    --model_configs.pna.epochs ${EPOCHS} \
    --model_configs.pna.batch_size ${BS} \
    --model_configs.pna.num_workers ${WK} \
    --trainer_stats pna_carbon \
    --trainer_stats_configs.codecarbon.run_num 1 \
    --trainer_stats_configs.codecarbon.project_name PNA \
    --trainer_stats_configs.codecarbon.output_dir '${REPO_DIR}/pna_result'
