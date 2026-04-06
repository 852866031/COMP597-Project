#!/bin/bash

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)
INITIAL_DIR=$(pwd)

cd ${REPO_DIR}

python3 launch.py $@

cd ${INITIAL_DIR}
