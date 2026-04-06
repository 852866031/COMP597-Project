
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "'${BASH_SOURCE[0]}' is a config file, it should not be executed. Source it to populate the variables."
    exit 1
fi

config_dir=$(readlink -f -n $(dirname ${BASH_SOURCE[0]}))

export COMP597_SLURM_CONFIG=${COMP597_SLURM_CONFIG:-${config_dir}/bash_slurm_config.sh}

. ${config_dir}/default_slurm_config.sh

scripts_dir=$(readlink -f -n $(dirname ${BASH_SOURCE[0]})/../scripts)

export COMP597_SLURM_CONFIG_LOG=false
export COMP597_SLURM_CPUS_PER_TASK=2
export COMP597_SLURM_MIN_MEM="1GB"
export COMP597_SLURM_NUM_GPUS=0
export COMP597_SLURM_JOB_SCRIPT=${scripts_dir}/bash_job.sh

unset scripts_dir
unset config_dir
