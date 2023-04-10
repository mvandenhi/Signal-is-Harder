#!/bin/bash

for state in {9..0}
do
    var1=$(date)
    seed="seed"${state}
    echo "${seed}"
	python make_dataset.py --dataset cifar10_type0 --random_state $state;
    cd ~/PATH/TO/LfF/
    source LfF_venv/bin/activate
    echo "-------------------------------------$var1-------------------------------------" >> results_LfF.txt

    python train.py with server_user corrupted_cifar10 type0 skewed1 severity4 $seed
    python train.py with server_user corrupted_cifar10 type0 skewed2 severity4 $seed
    python train.py with server_user corrupted_cifar10 type0 skewed3 severity4 $seed
    python train.py with server_user corrupted_cifar10 type0 skewed4 severity4 $seed
    python train.py with server_user corrupted_cifar10 type0 skewed5 severity4 $seed
    python train.py with server_user corrupted_cifar10 type0 skewed6 severity4 $seed
 
done
