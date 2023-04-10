#!/bin/bash

for state in {9..0}
do
    var1=$(date)
	python make_dataset.py --dataset cifar10_type0 --random_state $state --Dfa_dataset True;
    cd ~/PATH/TO/Learning-Debiased-Disentangled/
    source Dfa_venv/bin/activate
    echo "-------------------------------------$var1-------------------------------------" >> results_Dfa.txt
    # Cifar10c
    # 0.5pct
    python train.py --dataset cifar10c --exp=cifar10c_0.5_ours --lr=0.0005 --percent=0.5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb --use_resnet18_OURS True --seed=$state

    # 1pct
    python train.py --dataset cifar10c --exp=cifar10c_1_ours --lr=0.001 --percent=1pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=5 --lambda_swap_align=5 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb --use_resnet18_OURS True --seed=$state

    # 2pct
    python train.py --dataset cifar10c --exp=cifar10c_2_ours --lr=0.001 --percent=2pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=5 --lambda_swap_align=5 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb --use_resnet18_OURS True --seed=$state

    # 5pct
    python train.py --dataset cifar10c --exp=cifar10c_5_ours --lr=0.001 --percent=5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb --use_resnet18_OURS True --seed=$state

    # 10pct
    python train.py --dataset cifar10c --exp=cifar10c_10_ours --lr=0.001 --percent=10pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb --use_resnet18_OURS True --seed=$state

    # 20pct
    python train.py --dataset cifar10c --exp=cifar10c_20_ours --lr=0.001 --percent=20pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb --use_resnet18_OURS True --seed=$state

done