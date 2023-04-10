#!/bin/bash

for state in {9..0}
do
    var1=$(date)
    seed="seed"${state}
    echo "${seed}"
	python make_dataset.py --dataset colored_mnist --random_state $state;
    cd ~/PATH/TO/LfF/
    source LfF_venv/bin/activate
    echo "-------------------------------------$var1-------------------------------------" >> results_LfF.txt
    python train.py with server_user colored_mnist skewed5 severity4 $seed
    python train.py with server_user colored_mnist skewed6 severity4 $seed
    python train.py with server_user colored_mnist skewed1 severity4 $seed
    python train.py with server_user colored_mnist skewed2 severity4 $seed
    python train.py with server_user colored_mnist skewed3 severity4 $seed
    python train.py with server_user colored_mnist skewed4 severity4 $seed
 
done
