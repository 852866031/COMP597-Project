
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "'${BASH_SOURCE[0]}' a config file, it should not be executed. Source it to populate the variables."
    exit 1
fi

__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -ne 0 ]; then
	echo "Failed to obtain conda init hook."
	exit 1
fi
eval "$__conda_setup"
if [ $? -ne 0 ]; then
	echo "Failed to initialize conda."
	exit 1
fi
unset __conda_setup
