#!/bin/bash

for state in {9..0}
do
    var1=$(date)
    seed="seed"${state}
    echo "${seed}"
    
	python make_dataset.py --dataset cifar10_type0 --random_state $state;

    echo "-------------------------------------$var1-------------------------------------" >> results_Ours.txt

    python train.py --name "cifar10_type0_0.005pct" --bias_conflicting_perc 0.005 --severity 4 --dataset cifar10_type0 --model_tag ResNet_VAE --random_state $state  
    python train.py --name "cifar10_type0_0.01pct" --bias_conflicting_perc 0.01 --severity 4 --dataset cifar10_type0 --model_tag ResNet_VAE --random_state $state    
    python train.py --name "cifar10_type0_0.02pct" --bias_conflicting_perc 0.02 --severity 4 --dataset cifar10_type0 --model_tag ResNet_VAE --random_state $state    
    python train.py --name "cifar10_type0_0.05pct" --bias_conflicting_perc 0.05 --severity 4 --dataset cifar10_type0 --model_tag ResNet_VAE --random_state $state    
    python train.py --name "cifar10_type0_0.1pct" --bias_conflicting_perc 0.1 --severity 4 --dataset cifar10_type0 --model_tag ResNet_VAE --random_state $state  
    python train.py --name "cifar10_type0_0.2pct" --bias_conflicting_perc 0.2 --severity 4 --dataset cifar10_type0 --model_tag ResNet_VAE --random_state $state  

done