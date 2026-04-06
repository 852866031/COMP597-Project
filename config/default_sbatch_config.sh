
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "'${BASH_SOURCE[0]}' is a config file, it should not be executed. Source it to populate the variables."
    exit 1
fi

# The default config for sbatch relies on the default slurm configuration. 
# Please "default_slurm_config.sh" for further documentation.

config_dir=$(readlink -f -n $(dirname ${BASH_SOURCE[0]}))

. ${config_dir}/default_slurm_config.sh

unset config_dir

################################################################################
########################## Additional configurations ###########################
################################################################################

# The ouptut file where the logs of the job will be written.
# See --output in sbatch --help
export COMP597_SLURM_OUTPUT='comp597-%N-%j.log'
# The path to the directory where scripts are. Exceptionally, it needs to be 
# provided when using sbatch since the asynchronous nature of the execution 
# means the original script is copied to a different directory where the other 
# scripts will not be copied to. 
export COMP597_SLURM_SCRIPTS_DIR=$(readlink -f -n $(dirname ${BASH_SOURCE[0]})/../scripts)

################################################################################
################################################################################
################################################################################
