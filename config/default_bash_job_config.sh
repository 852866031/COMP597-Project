
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "'${BASH_SOURCE[0]}' is a config file, it should not be executed. Source it to populate the variables."
    exit 1
fi

# Values used to build some of the default configurations.
scripts_dir=$(readlink -f -n $(dirname ${BASH_SOURCE[0]})/../scripts)
config_dir=$(readlink -f -n $(dirname ${BASH_SOURCE[0]}))

export COMP597_JOB_CONFIG=${COMP597_JOB_CONFIG:-${config_dir}/bash_job_config.sh}

. ${config_dir}/default_job_config.sh

export COMP597_JOB_CONFIG_LOG=false

# Unset the variable as it is unused for bash jobs.
unset COMP597_JOB_COMMAND

# Unset the variables used locally.
unset scripts_dir
unset config_dir
