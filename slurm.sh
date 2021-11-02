#!/bin/bash
#SBATCH --job-name=l_ng_fix
#SBATCH --nodes=1                     # 1 node
#SBATCH --gres=gpu:1                  #--gres=gpu:2080:1 or --gres=gpu:1
#SBATCH --error=/nas/softechict-nas-2/avitto/DDsrl-Generation-Project-branch_gen_mod_fix_noise/log/2021-11-01_17-23-00-000000-legni_gan_cond_gen_modified_ol_0_fix_noise/slurm_err.txt               # error.txt standard error file
#SBATCH --output=/nas/softechict-nas-2/avitto/DDsrl-Generation-Project-branch_gen_mod_fix_noise/log/2021-11-01_17-23-00-000000-legni_gan_cond_gen_modified_ol_0_fix_noise/slurm_out.txt              # out.txt standard output file
#SBATCH --open-mode=append            #append-truncate
#SBATCH --array=0-2%1

conda activate env

#commands
# python test.py my-command

# cd ..
cd /homes/avitto/DDsrl-Generation-Project-branch_gen_mod_fix_noise
python digital_design.py train-gan --tag=2021-11-01_17-23-00-000000-legni_gan_cond_gen_modified_ol_0_fix_noise --yaml-file-path=legni.yaml

# sbatch slurm.sh