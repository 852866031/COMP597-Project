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

mv ${REPO_DIR}/src/trainer/stats/pna_carbon.py    ${REPO_DIR}/src/trainer/stats/pna_carbon.tmp
mv ${REPO_DIR}/src/trainer/stats/pna_manual_gc.py ${REPO_DIR}/src/trainer/stats/pna_manual_gc.tmp

# Restore files on exit, whether the job succeeds, fails, or is interrupted (Ctrl+C)
trap '
    mv ${REPO_DIR}/src/trainer/stats/pna_carbon.tmp    ${REPO_DIR}/src/trainer/stats/pna_carbon.py
    mv ${REPO_DIR}/src/trainer/stats/pna_manual_gc.tmp ${REPO_DIR}/src/trainer/stats/pna_manual_gc.py
' EXIT

### run PNA Simple Trainer
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model pna \
    --trainer pna_simple \
    --data pna_dataset \
    --model_configs.pna.epochs ${EPOCHS} \
    --model_configs.pna.batch_size ${BS} \
    --trainer_stats pna_simple \
    --trainer_stats_configs.codecarbon.run_num 1 \
    --trainer_stats_configs.codecarbon.project_name PNA \
    --trainer_stats_configs.codecarbon.output_dir '${REPO_DIR}/pna_result'
