
# This if-statement is not required, it is used to make sure no one accidently 
# tries to execute the config file.
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "'${BASH_SOURCE[0]}' is a config file, it should not be executed. Source it to populate the variables."
    exit 1
fi

# Override this to make sure the job runs on gpu-teach-03
export COMP597_SLURM_NODELIST="gpu-teach-03"

# Request more memory then the default:
export COMP597_SLURM_MIN_MEM="12GB"

# Any environment variable starting with "COMP597_SLURM_" in "default_config.sh" can be overriden as above except otherwise stated. 

