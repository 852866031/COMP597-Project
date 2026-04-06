
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "'${BASH_SOURCE[0]}' is a config file, it should not be executed. Source it to populate the variables."
    exit 1
fi

# The default config for srun simply relies on the default slurm configuration. 
# Please "default_slurm_config.sh" for further documentation.

config_dir=$(readlink -f -n $(dirname ${BASH_SOURCE[0]}))

. ${config_dir}/default_slurm_config.sh

unset config_dir
