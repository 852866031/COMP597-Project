#!/bin/bash

# Load config
. ${SETUP_SCRIPTS_DIR}/config.sh

mkdir -p ${COMP597_ADMIN_DIR}
setfacl -m g:${COMP597_USERS_GROUP}:--- ${COMP597_ADMIN_DIR}
setfacl -d -m g:${COMP597_USERS_GROUP}:--- ${COMP597_ADMIN_DIR}

cd ${COMP597_ADMIN_DIR}

git clone ${COMP597_REPO_URL}

mkdir -p ${COMP597_PIP_CACHE}

