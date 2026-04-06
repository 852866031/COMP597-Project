#!/bin/bash

# Required paths

SCRIPTS_DIR=${COMP597_SLURM_SCRIPTS_DIR:-$(readlink -f -n $(dirname $0))}
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

DEFAULT_CONFIG_FILE=${REPO_DIR}/config/default_bash_job_config.sh

# Load dependencies

. ${DEFAULT_CONFIG_FILE}

if [[ -f ${COMP597_JOB_CONFIG} ]]; then
	. ${COMP597_JOB_CONFIG}
fi

. ${SCRIPTS_DIR}/env_table.sh # Basics to print an environment variables table

# Set up the Conda environment

. ${SCRIPTS_DIR}/conda_init.sh

conda activate ${COMP597_JOB_CONDA_ENV_PREFIX}

# Logs

if [[ $COMP597_JOB_CONFIG_LOG = true ]]; then
	env_table "^SLURM_([CG]PU|MEM)" "SLURM Hardware Configuration"
	echo
	echo
	env_table "^SLURM_JOB" "SLURM Job Info"
	echo
	echo
	env_table "^COMP597_JOB_" "SLURM Job Configuration"
	echo
	echo
	env_table "^CONDA" "Conda Environment Variables"
	echo
	echo
fi

# Change the working directory if configured

if [[ -d "${COMP597_JOB_WORKING_DIRECTORY}" ]]; then
	cd ${COMP597_JOB_WORKING_DIRECTORY}
fi

# Run the job

eval "$@"
