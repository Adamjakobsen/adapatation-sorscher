#!/bin/bash
#ulimit -s unlimited
#SBATCH --job-name="ADAPT"
#SBATCH -p fpgaq #armq #milanq #fpgaq #milanq # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
##SBATCH --exclusive
##SBATCH --mem-per-cpu=128MB
#SBATCH --time=2-00:00
#SBATCH -o /home/mkkvalsu/slurm.column.%j.%N.out # STDOUT
#SBATCH -e /home/mkkvalsu/slurm.column.%j.%N.err # STDERR
##SBATCH --propagate=STACK


##ulimit -s 40960

module purge
module use /cm/shared/ex3-modules/latest/modulefiles
module load slurm/20.02.7
module load tensorflow2-extra-py37-cuda11.2-gcc8/2.5.2
# module load python3.7/numpy
#. /home/mkkvalsu/myenv/bin/activate
. /home/mkkvalsu/python_envs/grid_cells_env/bin/activate
#export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH



srun python3 /home/mkkvalsu/projects/adapatation-sorscher/main.py 
