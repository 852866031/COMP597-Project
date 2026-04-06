#!/bin/bash

export SETUP_SCRIPTS_DIR=$(readlink -f -n $(dirname $0))

# Set up admin directory

${SETUP_SCRIPTS_DIR}/admin_setup.sh

# Set up students' directory

${SETUP_SCRIPTS_DIR}/students_setup.sh

# Set up environment

${SETUP_SCRIPTS_DIR}/conda_setup.sh

