SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

### run PNA Spike Trainer (GC-on then GC-off epoch for spike attribution)
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model pna\
    --trainer pna_spike \
    --data pna_dataset \
    --model_configs.pna.epochs 5 \
    --model_configs.pna.batch_size 256 \
    --trainer_stats pna_spike \
    --trainer_stats_configs.codecarbon.run_num 1 \
    --trainer_stats_configs.codecarbon.project_name PNA \
    --trainer_stats_configs.codecarbon.output_dir '${REPO_DIR}/pna_result'
