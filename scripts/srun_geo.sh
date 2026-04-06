#!/bin/bash
# ./scripts/srun.sh --logging.level INFO --model dummy --data dummy

# Required paths
RED='\033[0;31m'
NC='\033[0m'  # No Color

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

DEFAULT_CONFIG_FILE=${REPO_DIR}/config/default_srun_config.sh

# Load dependencies

. ${DEFAULT_CONFIG_FILE}

if [[ -f ${COMP597_SLURM_CONFIG} ]]; then
	. ${COMP597_SLURM_CONFIG}
fi

. ${SCRIPTS_DIR}/env_table.sh # Basics to print an environment variables table

module load slurm # Required to have access to SLURM commands.

# Logs

if [[ $COMP597_SLURM_CONFIG_LOG = true ]]; then
	env_table "^COMP597_SLURM_" "SLURM Configuration"
	echo
	echo
fi

# Launch the job
COMP597_SLURM_JOB_SCRIPT=/home/2020/jchen213/milabench/benchmarks/geo_gnn/job_geo_gnn.sh
echo -e "${RED}Launching SLURM job script:${NC} ${RED}${COMP597_SLURM_JOB_SCRIPT}${NC}"
srun \
	--partition=${COMP597_SLURM_PARTITION} \
	--mem=${COMP597_SLURM_MIN_MEM} \
	--time=${COMP597_SLURM_TIME_LIMIT} \
	--ntasks=${COMP597_SLURM_NTASKS} \
	--account=${COMP597_SLURM_ACCOUNT} \
	--nodelist=${COMP597_SLURM_NODELIST} \
	--cpus-per-task=${COMP597_SLURM_CPUS_PER_TASK} \
	--qos=${COMP597_SLURM_QOS} \
	--gpus=${COMP597_SLURM_NUM_GPUS} \
	${COMP597_SLURM_JOB_SCRIPT} $@
