SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

### run PNA Measurement Trainer with manual-GC timing stats
### Uses PNACarbonTrainer (--trainer pna): GC is disabled during training and
### gen-2 collections are forced between epochs to keep pauses outside measured
### steps.  Step latency is recorded by PNAManualGCStats (--trainer_stats
### pna_manual_gc) which writes to pna_result/manual/.
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model pna \
    --trainer pna \
    --data pna_dataset \
    --model_configs.pna.epochs 5 \
    --model_configs.pna.batch_size 256 \
    --trainer_stats pna_manual_gc \
    --trainer_stats_configs.codecarbon.run_num 1 \
    --trainer_stats_configs.codecarbon.project_name PNA \
    --trainer_stats_configs.codecarbon.output_dir '${REPO_DIR}/pna_result'
