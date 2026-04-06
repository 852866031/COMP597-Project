#!/bin/bash

# Load config
. ${SETUP_SCRIPTS_DIR}/config.sh

mkdir -p ${COMP597_STUDENTS_DIR}
setfacl -m g:${COMP597_USERS_GROUP}:rwx ${COMP597_STUDENTS_DIR}
setfacl -d -m g:${COMP597_USERS_GROUP}:r-x ${COMP597_STUDENTS_DIR}
setfacl -d -m g:${COMP597_ADMIN_GROUP}:rwx ${COMP597_STUDENTS_DIR}

