# Slurm

The experiments require access to GPUs to train machine learning models. We have obtained access to the SOCS GPU nodes for all students in the course. If you do not have correct access to SLURM (i.e. you cannot run the GPT2 tutorial in the assignment text out of the box), please contact the course staff via Ed. 

Jobs can only be run on those nodes using [Slurm](https://slurm.schedmd.com/documentation.html). "GPU node" and "Slurm node" are used interchangeably in this file.

Additionally, please find below a diagram overview of the systems used in this course as a visual aid to the contents of this file.

![image](./COMP597-system-overview.png)

## Accessing the GPU nodes

McGill IT has documentation on accessing the GPU nodes. You can find it [here](https://docs.sci.mcgill.ca/COMP/slurm/). The account for this course is `energy-efficiency-comp597` and the QOS is `comp597`. 

| Category | Information |
| :--- | :--- |
| Account | `energy-efficiency-comp597` |
| QOS | `comp597` |
| Max CPU cores per task | 4 |
| Max GPUs per job | 2 |
| Max memory per job | 16GB |
| Max time limit | 6h |

## Storage

There are three main disk storage locations that can be used for the project. The details of each are below. Please read them in detail, as this can have an impact on the project.

### Home directory

Your user's home directory is mounted on all the GPU nodes at its habitual location (same as on mimi). It is a good place to store your code and configuration files, as it is easily accessible. However, keep in mind that you are only allocated a small amount of disk space, which can fill up very quickly. You can see your current disk usage and the maximum you are allowed to use using the `quota` command.

### Slurm node local storage

All the Slurm nodes have a local partition dedicated to storing data when running jobs. Everything you create in that storage partition is persistent, and can outlive your job. However, McGill IT can wipe the storage at any moment (when there are no jobs running on the Slurm node), so you should make backups of what you do not want to lose. At a minimum, we recommend you save all the code to git to make sure you don't lose your work. Typically, they clean up the storage partition at the end of every semester or when it gets full, but there are no guarantees. 

The storage partitions are quite large (up to 7TB), but are local to each Slurm node. It would be a good place to store a Conda environment or a downloaded pre-trained machine learning model, as they can be easily created again if your data gets deleted.

Typically, the path to this storage partition is `/mnt/teaching/slurm`, and should be the same on any Slurm node. The scripts we provide already set it in an environment variable (see `config/default_job_config.sh`)

### COMP597 shared storage

We have obtained a shared storage partition that is mounted on all the Slurm nodes. All students registered in the course should have access to it at the path `/home/slurm/comp597` (let us know on Ed if you don't). See the basic structure below:

```
/home/slurm/comp597
├── admin 
├── conda
│   └── envs
│       └── comp597
├── example
│   └── c4
└── students
    └── ...
```

The `admin` directory is used by the course staff. You do not need to worry about it.

The `example` directory contains data used to run the provided GPT2 example.

The course administrators manage and maintain a Conda environment that should contain everything you need for the project. It can be used with `conda activate /home/slurm/comp597/conda/envs/comp597`. If any modules are missing, feel free to contact us on Ed, and we will install it. 

Finally, the `students` directory is where you can create additional content. To keep it organized, we ask that you create your own directory under `students` and work from there.

This storage partition is the perfect space to store your dataset, as it will be mounted on every Slurm node. 

We ask that you be mindful of your storage usage on this partition. The partition is only 200GB in size, which would accommodate 10GB per team with some spare room for additional overhead. You can check the disk usage at a given path using `du -h SOME_PATH`. 

## Running a job

Slurm provides two main tools to run jobs: [`srun`](https://slurm.schedmd.com/srun.html) and [`sbatch`](https://slurm.schedmd.com/sbatch.html). We provide you with scripts that wrap around these tools and manage a lot of the configurations for you, with the option to extend the configurations. Essentially, the difference between `srun` and `sbatch` is that the former will run on your terminal and the process will exit only when the job will have completed, while the latter will schedule the job and exit immediately. 

We provide the script `scripts/srun.sh` that wraps around the `srun` tool. The script uses the default config located in `config/default_srun_config.sh`. For `sbatch`, we provide `scripts/sbatch.sh`, which uses the default config located in `config/default_sbatch_config.sh`. Both wrapper's default configuration files contain the documentation to configure them.

For both wrapper scripts, additional configurations can be added by creating a bash file named `config/slurm_config.sh`, by setting `COMP597_SLURM_CONFIG` to the path to your config. For more details, see `config/default_slurm_config.sh`.

Both wrappers provided will run the `scripts/job.sh` script by default. This script loads the default `config/default_job_config.sh` by default, which also contains the configuration documentation. The wrappers will execute on the computer you are using, whereas this job script will execute on the Slurm node. By default, the `scripts/job.sh` script is designed to launch the code in this repository using the `scripts/launch.sh` script. This means that `scripts/srun.sh` will run `scripts/launch.sh` which will run `python3 launch.py` on the Slurm node by default. 

All arguments provided to either wrapper are transparently given to the job script, so `srun.sh --model gpt2` will give the arguments `--model gpt2` to the job, which by default would run as `job.sh --model gpt2`, and since `job.sh` gives all its arguments to the command it needs to execute, then it would run `python3 launch.py --model gpt2`. Note that the `job.sh` script evaluates the inputs, so if you do `./job.sh --logging.filename '${COMP597_JOB_STUDENTS_BASE_DIR}'/launch.log`, then `${COMP597_JOB_STUDENTS_BASE_DIR}` will get evaluated on the Slurm node, which would result in running something along the lines of `python3 --logging.filename /home/slurm/comp597/students/launch.log`.

Additionally, we provide the script `scripts/bash_srun.sh` that allows you to run bash commands on the slurm nodes by running scripts through slurm. Essentially, this scripts will take all provided inputs and run them on the configured Slurm node using the `eval` bash built-in. You can try `./scripts/bash_srun.sh "cd /home/slurm/comp597; tree -d -L 2"` as example.
