#!/bin/bash



#SBATCH --nodes=1  
#SBATCH --ntasks=1  
#SBATCH --job-name=Hnet   
#SBATCH --time=14:00:00  
#SBATCH --partition=batch  
#SBATCH --gres=gpu:a100:1
#SBATCH --reservation=A100
#SBATCH --mem=12768M






eval "$(conda shell.bash hook)"
# Change to the name to your conda env.
module load cuda/11.5.2
conda activate PyG 
# python ray_tune_RBMP.py  --dataset 'cornell' --name 'cornell_10fixed_t2'  --epoch 100  --time 2
# python ray_tune_RBMP.py  --dataset 'cornell' --name 'cornell_10fixed_t4'  --epoch 100  --time 4
# python ray_tune_RBMP.py  --dataset 'cornell' --name 'cornell_10fixed_t6'  --epoch 100  --time 6
python ray_tune_RBMP.py  --dataset 'Cora' --name 'cora_10fixed_1'  --epoch 50
# python ray_tune_RBMP.py  --dataset 'cornell' --name 'cornell_10fixed_t10'  --epoch 100  --time 10
# python ray_tune_RBMP.py  --dataset 'wisconsin' --name 'wisconsin_10fixed_1'  --epoch 100  

# python ray_tune_RBMP.py  --dataset 'chameleon' --name 'chameleon_10fixed_1'  --epoch 100 
# python ray_tune_RBMP.py  --dataset 'squirrel' --name 'squirrel_10fixed_1'  --epoch 100
# python ray_tune_RBMP.py  --dataset 'CoauthorCS' --name 'CoauthorCS_10fixed_2'  --epoch 200 
