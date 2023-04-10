#!/bin/bash

for state in {9..0}
do
    var1=$(date)
	python make_dataset.py --dataset colored_mnist --random_state $state --Dfa_dataset True;
    cd ~/PATH/TO/Learning-Debiased-Disentangled/
    source Dfa_venv/bin/activate
    echo "-------------------------------------$var1-------------------------------------" >> results_Dfa.txt
    # CMNIST
    # 20pct
    python train.py --dataset cmnist --exp=cmnist_20_ours --lr=0.01 --percent=20pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb --seed=$state 

    # 10pct
    python train.py --dataset cmnist --exp=cmnist_10_ours --lr=0.01 --percent=10pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb --seed=$state 

    # 0.5pct
    python train.py --dataset cmnist --exp=cmnist_0.5_ours --lr=0.01 --percent=0.5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb --seed=$state

    # 1pct
    python train.py --dataset cmnist --exp=cmnist_1_ours --lr=0.01 --percent=1pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb --seed=$state 

    # 2pct
    python train.py --dataset cmnist --exp=cmnist_2_ours --lr=0.01 --percent=2pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb --seed=$state

    # 5pct
    python train.py --dataset cmnist --exp=cmnist_5_ours --lr=0.01 --percent=5pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb --seed=$state 
done