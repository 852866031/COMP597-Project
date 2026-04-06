#!/bin/bash

# Load config
. ${SETUP_SCRIPTS_DIR}/config.sh

mkdir -p ${COMP597_CONDA_DIR}
setfacl -m g:${COMP597_USERS_GROUP}:r-x ${COMP597_CONDA_DIR}
setfacl -d -m g:${COMP597_USERS_GROUP}:r-x ${COMP597_CONDA_DIR}
setfacl -m g:${COMP597_ADMIN_GROUP}:rwx ${COMP597_CONDA_DIR}
setfacl -d -m g:${COMP597_ADMIN_GROUP}:rwx ${COMP597_CONDA_DIR}

export PIP_CACHE_DIR=${COMP597_PIP_CACHE}

conda create --prefix=${COMP597_CONDA_ENV_PREFIX} python=${COMP597_CONDA_ENV_PYTHON_VERSION}

. ${SETUP_SCRIPTS_DIR}/../scripts/conda_init.sh
conda activate ${COMP597_CONDA_ENV_PREFIX}

requirements_file=${SETUP_SCRIPTS_DIR}/../requirements.txt

pip install -r ${requirements_file}

rm -rf ${COMP597_PIP_CACHE}/*

