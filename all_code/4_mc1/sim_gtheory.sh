#!/bin/bash
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=janwillemnijenhuis@gmail.com

#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --mem=90G

#Loading modules
module load 2020
module load Python/3.8.2-GCCcore-9.3.0
pip3 install --user pandas
pip3 install --user pickle
pip3 install --user tqdm
pip3 install --user openpyxl

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
for i in `seq 1 10`; do
	python $HOME/sim_g.py &
	
done
wait
	
