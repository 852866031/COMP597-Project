#!/bin/bash

function usage() {
	echo "Usage: ./env_setup.sh PATH"
}

function error() {
	echo "[ERROR] $1"
	usage
	exit 1
}

function join_path() {
	p1=$1
	p2=$2
	echo "${p1}/${p2}" | sed -e 's/\/\/*/\//g'
}

if [[ $# -eq 0 ]]; then
	error "Please provide a path to store the environment and cache data"
elif [[ $# -ne 1 ]]; then 
	error "Too many arguments provided"
elif [[ ! -d $1 ]]; then
	error "The provided path '$1' does not exist."
fi

export BASE_STORAGE_PATH=$1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${SCRIPT_DIR}

# Step 1
mkdir -p $(join_path "${BASE_STORAGE_PATH}" "cache")
export PIP_CACHE_DIR=$(join_path "${BASE_STORAGE_PATH}" "cache/pip")
export HF_HOME=$(join_path "${BASE_STORAGE_PATH}" "cache/huggingface")

# Step 2
conda init bash

# Step 3
mkdir -p $(join_path "${BASE_STORAGE_PATH}" "conda/envs")
ENV_PATH=$(join_path "${BASE_STORAGE_PATH}" "conda/envs/COMP597-project")
conda create --prefix ${ENV_PATH} python=3.14

# Step 4
conda activate ${ENV_PATH}

# Step 5
pip install -r requirements.txt

# Step 6 
cat >local_env.sh <<EOF
#!/bin/bash 

. ~/.bashrc

export PIP_CACHE_DIR=${PIP_CACHE_DIR}
export HF_HOME=${HF_HOME}
conda activate ${ENV_PATH}
EOF
