#!/bin/bash

# Required paths

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

DEFAULT_CONFIG_FILE=${REPO_DIR}/config/default_sbatch_config.sh

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

sbatch \
	--output=${COMP597_SLURM_OUTPUT} \
	--partition=${COMP597_SLURM_PARTITION} \
	--mem=${COMP597_SLURM_MIN_MEM} \
	--time=${COMP597_SLURM_TIME_LIMIT} \
	--ntasks=${COMP597_SLURM_NTASKS} \
	--account=${COMP597_SLURM_ACCOUNT} \
	--nodelist=${COMP597_SLURM_NODELIST} \
	--cpus-per-task=${COMP597_SLURM_CPUS_PER_TASK} \
	--qos=${COMP597_SLURM_QOS} \
	--gpus=${COMP597_SLURM_NUM_GPUS} \
	--export=COMP597_SLURM_SCRIPTS_DIR=${COMP597_SLURM_SCRIPTS_DIR} \
	${COMP597_SLURM_JOB_SCRIPT} $@
