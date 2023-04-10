#!/bin/bash

for state in {9..0}
do
    var1=$(date)
    seed="seed"${state}
    echo "${seed}"
	python make_dataset.py --dataset colored_mnist --random_state $state;
    echo "-------------------------------------$var1-------------------------------------" >> results_vanilla.txt
    python train_vanilla.py --name "CMNIST_0.005pct" --bias_conflicting_perc 0.005 --severity 4 --dataset colored_mnist --model_tag MLP_VAE --results_filename results_vanilla --epochs 100 --random_state $state 
    python train_vanilla.py --name "CMNIST_0.01pct" --bias_conflicting_perc 0.01 --severity 4 --dataset colored_mnist --model_tag MLP_VAE --results_filename results_vanilla --epochs 100 --random_state $state   
    python train_vanilla.py --name "CMNIST_0.02pct" --bias_conflicting_perc 0.02 --severity 4 --dataset colored_mnist --model_tag MLP_VAE --results_filename results_vanilla --epochs 100 --random_state $state   
    python train_vanilla.py --name "CMNIST_0.05pct" --bias_conflicting_perc 0.05 --severity 4 --dataset colored_mnist --model_tag MLP_VAE --results_filename results_vanilla --epochs 100 --random_state $state   
    python train_vanilla.py --name "CMNIST_0.1pct" --bias_conflicting_perc 0.1 --severity 4 --dataset colored_mnist --model_tag MLP_VAE --results_filename results_vanilla --epochs 100 --random_state $state   
    python train_vanilla.py --name "CMNIST_0.2pct" --bias_conflicting_perc 0.2 --severity 4 --dataset colored_mnist --model_tag MLP_VAE --results_filename results_vanilla --epochs 100 --random_state $state   

 
done
