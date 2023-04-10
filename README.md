# Signal Is Harder To Learn Than Bias
This is the code for the ICLR 2023 Workshop paper:
[Signal Is Harder To Learn Than Bias: Debiasing with Focal Loss](https://openreview.net/forum?id=PEPSFxkjqg)

For running the code:
1. Create a new environment
2. Execute ```pip install -r requirements.txt```
3. Adapt the data directories in the config.yaml file to your own path.
4. For Weights&Biases support, in config.py set 'wandb_logging' to True and in train.py adapt wandb entity to yours
5. Run ```bash scripts/run_cmnist_Ours.sh``` or ```bash scripts/run_cifar10c_Ours.sh```


## Citation
If you find our work is helpful for your research, please cite our paper.
```
@inproceedings{
vandenhirtz2023signalisharder,
title={Signal Is Harder To Learn Than Bias: Debiasing with Focal Loss},
author={Moritz Vandenhirtz and Laura Manduchi and Ri{\v{c}}ards Marcinkevi{\v{c}}s and Julia E Vogt},
booktitle={ICLR 2023 Workshop on Domain Generalization (DG)},
year={2023},
url={https://openreview.net/forum?id=PEPSFxkjqg}
}
```