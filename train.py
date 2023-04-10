import os
import yaml
import argparse
import wandb
import time
import sys
from tqdm import tqdm
import numpy as np
from uuid import uuid4

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import normalize
from data.util import get_dataset, IdxDataset, IdxDataset2
from module.util import get_model
from util import set_seed, get_optimizer, evaluate, \
    GeneralizedCELoss, evaluate_batch, save_img, save_img_adv, \
    EarlyStopping, early_stop, capture_dataset, bias_visualization



def main():

    # configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # manual overwriting of configuration for scripts
    #   initialize parser
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None, help = "Name of experiment")
    parser.add_argument("--bias_conflicting_perc", default=None, type=float, help = "Percentage of bias conflicting samples in dataset")
    parser.add_argument("--severity", default=None, type=int, help = "Severity of bias")
    parser.add_argument("--dataset", default=None, help = "Choice of dataset")
    parser.add_argument("--model_tag", default=None, help = "Choice of model")
    parser.add_argument("--q", default=None, type=float, help = "q for GCE loss")
    parser.add_argument("--random_state", default=None, type=int, help="Random state for seed")
    parser.add_argument("--results_filename", default=None, help="Name of file to store results")
    parser.add_argument("--VAE_weight", default=None, type=float, help="Weight of KL&Reconstruction loss")
    parser.add_argument("--reconst_weight", default=None, type=float, help="Weight of Reconstruction loss")
    args = parser.parse_args()
    # Replace all specified arguments
    updateable = [config["name"],config["data"]["bias_conflicting_perc"],config["data"]["severity"],config["data"]["dataset"],config["model"]["tag"],config["loss"]["GCE_q"],config["random_state"],config["results_filename"],config["loss"]["VAE_weight"],config["loss"]["reconst_weight"]]
    values = []
    for i,v in enumerate(vars(args).values()):
        if v != None:
             values.append(v)
             print("Overwriting configuration")
        else: values.append(updateable[i])
    config["name"],config["data"]["bias_conflicting_perc"],config["data"]["severity"],config["data"]["dataset"],config["model"]["tag"],config["loss"]["GCE_q"],config["random_state"],config["results_filename"],config["loss"]["VAE_weight"],config["loss"]["reconst_weight"] = values
    # configuration sanity check
    if not (
        (config["data"]["dataset"] == "colored_mnist" and config["model"]["tag"] == "MLP") or
        (config["data"]["dataset"] == "colored_mnist" and config["model"]["tag"] == "MLP_VAE") or
        (config["data"]["dataset"] == "cifar10_type0" and config["model"]["tag"] == "ResNet20") or
        (config["data"]["dataset"] == "cifar10_type1" and config["model"]["tag"] == "ResNet20") or 
        (config["data"]["dataset"] == "cifar10_type1" and config["model"]["tag"] == "ResNet20") or
        (config["data"]["dataset"] == "cifar10_type0" and config["model"]["tag"] == "ResNet_VAE") or
        (config["data"]["dataset"] == "cifar10_type1" and config["model"]["tag"] == "ResNet_VAE") or
        (config["data"]["dataset"] == "camelyon17_type0" and config["model"]["tag"] == "ResNet_VAE") or
        (config["data"]["dataset"] == "camelyon17_type1" and config["model"]["tag"] == "ResNet_VAE") 
        ):
        print("Are you sure you want to use the dataset "+config["data"]["dataset"]+" with the model "+ config["model"]["tag"]+"?")

    # define variables from config
    batch_size = config["train"]["batch_size"]
    epochs = config["train"]["epochs"]
    random_state = config["random_state"]

    # wandb support
    mode = "online" if config['wandb_logging'] else "disabled"
    #wandb.login(key="e34806ecc80c88cfb408eda2e5848fa494272f15")
    wandb.init(
        project="Signalisharder", 
        entity="username", 
        config=config, 
        mode=mode
    )
    wandb.run.name = wandb.run.name.split("-")[-1] + "-"+config['name'] 
    #wandb.run.save()

    print("Running experiment: {}".format(config["name"]))
    # set seed
    set_seed(random_state)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # load dataset
    train_dataset = get_dataset(
        config,
        dataset_split="train"
    )
    
    test_dataset = get_dataset(
        config,
        dataset_split="eval"
    )

    # Adapt train dataset twice to get indices of subset as well as full dataset
    train_dataset = IdxDataset(train_dataset)
    test_dataset = IdxDataset(test_dataset)
    train_dataset_splitted = IdxDataset2(train_dataset)


    train_loader = DataLoader(
        train_dataset_splitted,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )            

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     #num_workers=16,
    #     pin_memory=True,
    # )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    # define signal and bias model and model for approximating disentaglement loss
    model_s = get_model(config).to(device)
    model_b = get_model(config).to(device)
    # Decoder of bias network not used
    for p in model_b.decoder.parameters():
        p.requires_grad = False

    # define optimizer
    optimizer_s = get_optimizer(model_s, config)
    optimizer_b = get_optimizer(model_b, config)

    # define scheduler
    if config["model"]["tag"] == "MLP_VAE":
        patience = config["early_stop"]["patience_MLP"]
    elif config["model"]["tag"] == "ResNet_VAE":
        patience = config["early_stop"]["patience_ResNet"]
    else: raise NotImplementedError("Patience")
    scheduler_s = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s, verbose=True, patience = int(patience/2)-1, factor=config["optimizer"]["lr_gamma"],threshold=0.00001)
    scheduler_b = optim.lr_scheduler.ReduceLROnPlateau(optimizer_b, verbose=True, patience = int(patience/2)-1, factor=config["optimizer"]["lr_gamma"],threshold=0.00001)

    # define loss function
    criterion_s = nn.CrossEntropyLoss(reduction='none')
    criterion_b = GeneralizedCELoss(config)

    # early stopping
    os.makedirs("./saved_models/ours_s/", exist_ok=True)
    os.makedirs("./saved_models/ours_b/", exist_ok=True)
    timestamp = time.strftime(' %d-%b-%Y_%H:%M', time.localtime())
    id = str(uuid4())
    save_path_s = "./saved_models/ours_s/" + config["name"] + timestamp + id + ".pt"
    save_path_b = "./saved_models/ours_b/" + config["name"] + timestamp + id + ".pt"
    stopping_criteria_vae = EarlyStopping(patience=patience, verbose=True, path_s = save_path_s, path_b = save_path_b, delta=0.00001) # Same as scheduler
    early_stop_v = False

    # training & validation & test
    for epoch in range(epochs):
        if early_stop_v == False:
            train_dataset_splitted.make_train()
            train(model_s, model_b, train_loader, early_stop_v, optimizer_s, optimizer_b, criterion_s, criterion_b, epoch, epochs, device, config)
            train_dataset_splitted.make_biased_val() # Biased validation set to determine early stopping of vae
            early_stop_v = early_stop(model_s, model_b, train_loader, stopping_criteria_vae, scheduler_s, scheduler_b, epoch, device, config)
            if early_stop_v == True: # Reverse models to early stopping points 
                model_s.load_state_dict(torch.load(save_path_s, map_location=device))
                model_b.load_state_dict(torch.load(save_path_b, map_location=device))
                for p in model_s.encoder.parameters(): # Freeze signal part of VAE.
                    p.requires_grad = False
                for p in model_s.decoder.parameters():
                    p.requires_grad = False
                for p in model_b.parameters():
                    p.requires_grad = False
                train_dataset_splitted.make_biased_val()
        else: break
        validate(model_s, model_b, test_loader, epoch, device, config) # Important: Until having made a decision we use the test set as validation set for model analysis!
    #capture_dataset(train_loader, config)
    test_acc_s, test_acc_b = test(model_s, model_b, test_loader, epochs, device, config)
    # train_reconst_loss = get_reconst_loss(model_s, model_b, train_loader, device, config, mode = "train")
    # test_reconst_loss = get_reconst_loss(model_s, model_b, test_loader, device, config, mode = "test")
    # Saving result & Checkpoint
    with open(config["results_filename"]+'.txt', 'a') as f:
        f.writelines((['{} signal: {:8.4f}\n'.format(config["name"], test_acc_s)]))
        # f.writelines((['{} biased: {:8.4f}\n'.format(config["name"], test_acc_b)]))
        # f.writelines((['{} train_reconst: {:8.4f}\n'.format(config["name"], train_reconst_loss)]))
        # f.writelines((['{} test_reconst: {:8.4f}\n'.format(config["name"], test_reconst_loss)]))

    
    # Save images to wandb
    save_img(model_s, model_b, test_loader, config, device)
    for i in range(5):
        save_img_adv(model_s, model_b, test_loader, epoch, config, device, training=False)
        bias_visualization(model_s, model_b, train_loader, config, device) #Using biased validation set to have (enough) bias-aliged images
    wandb.finish(quiet=True)
    os.remove(save_path_s)
    os.remove(save_path_b)

def train(
    model_s,
    model_b,
    train_loader,
    early_stop_v,
    optimizer_s,
    optimizer_b,
    criterion_s,
    criterion_b,
    epoch,
    epochs,
    device,
    config
):
    """Main training loop, where the network is trained
    Args:
    UPDATE ARG DESCRIPTION
        model: baseline model
        train_loader: loader with the training data
        optimizer: optimizer for backpropagation
        criterion: loss function
        epoch:  current epoch
        epochs: max number of epochs
        device: current device (cpu or gpu)
    """
    train_loader = tqdm(train_loader, position=0, leave=False)
    train_loader.set_description(f"Epoch [{epoch}/{epochs}]")

    total_corr_aligned_s, total_corr_conflicting_s, total_corr_s, total_count_aligned, total_count_conflicting, total_count = 0, 0, 0, 0, 0, 0
    total_corr_aligned_b, total_corr_conflicting_b, total_corr_b, = 0, 0, 0
    total_corr_aligned_s_adv, total_corr_conflicting_s_adv, total_corr_s_adv, = 0, 0, 0
    # training loop
    model_s.train()
    model_b.train()
    if early_stop_v:
        model_b.eval()
    for idx, (subset_idx, full_idx, data, attr) in enumerate(train_loader):
        data, attr = data.to(device), attr.to(device)
        label = attr[:, 0] # Assuming label is in first column and bias in second of variable attr!
        # bias = attr[:, 1]

        # Getting predictions
        z_s, logits_s, mean_s, logvar_s = model_s(data)
        z_b, logits_b, mean_b, logvar_b = model_b(data)
        # z_s_avging_pos = z_s[label.bool()][torch.randperm(sum(label==1))]
        # z_s_avging_neg = z_s[~label.bool()][torch.randperm(sum(label==0))]
        # z_s[label.bool()] = (z_s[label.bool()] + z_s_avging_pos.detach())/2 ###Detach yes/no?
        # z_s[~label.bool()] = (z_s[~label.bool()] + z_s_avging_neg.detach())/2 ###Detach yes/no?
        # logits_s = model_s.predict(z_s)
        z = torch.cat((z_s, z_b), dim=1)
        mean = torch.cat((mean_s, mean_b), dim=1)
        logvar = torch.cat((logvar_s, logvar_b), dim=1)
        x_reconst = model_s.reconstruct(z)

        # VAE losses     
        # Compute reconstruction loss and kl divergence for both encoders together
        # Sum over dimensions, average over batch to have loss weighting hyperparameters being independent of batch size
        if (config["data"]["dataset"] == "cifar10_type0" or config["data"]["dataset"] == "cifar10_type1"): # Backtransform preprocessing standardization for CE
            data_backtransformed = normalize(data,-np.divide([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]), np.divide([1,1,1],[0.2023, 0.1994, 0.2010]))
            reconst_loss = F.mse_loss(x_reconst, data_backtransformed, reduction='none').sum(dim=(1,2,3))
        elif config["data"]["dataset"] == "colored_mnist":
            reconst_loss = F.binary_cross_entropy(x_reconst, data, reduction='none').sum(dim=(1,2,3))
        elif (config["data"]["dataset"] == "camelyon17_type0" or config["data"]["dataset"] == "camelyon17_type1" or config["data"]["dataset"] == "camelyon17_type2"):
            reconst_loss = F.mse_loss(x_reconst, data, reduction='none').sum(dim=(1,2,3))
        else: raise NotImplementedError("reconst_loss")
        kl_div = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(),dim=1)

        # Rescaling both VAE losses in order to be invariant to image resolution in hyperparametertuning.
        reconst_loss /= x_reconst.view(len(x_reconst),-1).shape[1]
        kl_div /= x_reconst.view(len(x_reconst),-1).shape[1]

        reconst_loss *=config["loss"]["reconst_weight"]

        # 1-yhat_b instead of RDS
        prob_b = F.softmax(logits_b, dim=1)
        if np.isnan(prob_b.mean().item()):
            raise NameError("prob_b")
        y_hat_b = torch.gather(prob_b, 1, torch.unsqueeze(label, 1)).squeeze().detach().cpu()
        if np.isnan(y_hat_b.mean().item()):
            raise NameError("y_hat_b")


        loss_weight = (1-y_hat_b)**config["loss"]["GCE_q"] # 1-yhat for hard-to-learn samples. 
        rel_diff_score = loss_weight.detach().to(device)
        
        # Calculate and weigh classifier losses
        loss_indiv_s = criterion_s(logits_s,label)* rel_diff_score
        loss_indiv_b = criterion_b(logits_b,label)

        # Evaluate metrics for logging and backpropagating
        corr_aligned_s, corr_conflicting_s, corr_s, loss_aligned_s, loss_conflicting_s, loss_s, aligned_len, conflicting_len, batch_len = evaluate_batch(logits_s,attr,loss_indiv_s)
        corr_aligned_b, corr_conflicting_b, corr_b, loss_aligned_b, loss_conflicting_b, loss_b = evaluate_batch(logits_b,attr,loss_indiv_b)[0:6]

        if torch.isnan(loss_s): 
            raise NameError('loss_s_update')
        if torch.isnan(loss_b):
            raise NameError('loss_b_update')
        # Backprop model
        optimizer_s.zero_grad()
        optimizer_b.zero_grad()
        loss = loss_s + loss_b + reconst_loss.mean() + kl_div.mean()
        loss.backward()
        optimizer_s.step()
        optimizer_b.step()

        # Calculate metrics for logging
        total_corr_aligned_s += corr_aligned_s
        total_corr_conflicting_s += corr_conflicting_s
        total_corr_s += corr_s
        total_corr_aligned_b += corr_aligned_b
        total_corr_conflicting_b += corr_conflicting_b
        total_corr_b += corr_b
        total_count_aligned += aligned_len
        total_count_conflicting += conflicting_len
        total_count += batch_len

        train_loader.set_postfix({"loss_s": "{:.3f}".format(loss_s.item()), "loss_b": "{:.3f}".format(loss_b.item()), 
                                  "acc_s": "{:.3f}".format(corr_s.item() / batch_len), "acc_b": "{:.3f}".format(corr_b.item() / batch_len)})
        wandb.log({"loss_s": loss_s, "loss_s_align": loss_aligned_s, "loss_s_conflict": loss_conflicting_s, "reconstruction_loss": reconst_loss.mean()})
        wandb.log({"loss_b": loss_b, "loss_b_align": loss_aligned_b, "loss_b_conflict": loss_conflicting_b, "loss": loss})

    if config["wandb_logging"]:
        save_img_adv(model_s, model_b, train_loader, epoch, config, device, training=True)
        wandb.log({"acc_s_train": total_corr_s / total_count, "acc_s_train_align": total_corr_aligned_s / total_count_aligned, 
            "acc_s_train_conflict": total_corr_conflicting_s / total_count_conflicting, "epoch": epoch})
        wandb.log({"acc_b_train": total_corr_b / total_count, "acc_b_train_align": total_corr_aligned_b / total_count_aligned, 
            "acc_b_train_conflict": total_corr_conflicting_b / total_count_conflicting, "epoch": epoch})
        wandb.log({"acc_s_train_adv": total_corr_s_adv / total_count, "acc_s_train_align_adv": total_corr_aligned_s_adv / total_count_aligned, 
            "acc_s_train_conflict_adv": total_corr_conflicting_s_adv / total_count_conflicting, "epoch": epoch})
    print(
        "| epoch {:3d} | training accuracy_biased {:8.3f}".format(
            epoch, total_corr_b / total_count
        )
    )


def validate(model_s, model_b, val_loader, epoch, device, config):
    """Main test loop, where the network is tested in the end
    Args:
        model: our pytorch model
        val_loader: loader with the validation data
        device: current device (cpu or gpu)
    """
    # testing the model
    model_s.eval()
    model_b.eval()
    val_acc_aligned_s, val_acc_conflicting_s, val_acc_s = evaluate(model_s, val_loader, device)
    val_acc_aligned_b, val_acc_conflicting_b, val_acc_b = evaluate(model_b, val_loader, device)
    if config["loss"]["perturbation"]:
        save_img_adv(model_s, model_b, val_loader, epoch, config, device)
    wandb.log({"acc_s_val": val_acc_s, "acc_s_val_align": val_acc_aligned_s, "acc_s_val_conflict": val_acc_conflicting_s, "epoch": epoch})
    wandb.log({"acc_b_val": val_acc_b, "acc_b_val_align": val_acc_aligned_b, "acc_b_val_conflict": val_acc_conflicting_b, "epoch": epoch})
    print("validation accuracy of unbiased model {:8.3f}".format(val_acc_s))
    print("validation accuracy of biased model {:8.3f}".format(val_acc_b))


def test(model_s, model_b, test_loader, epochs, device, config):
    """Main test loop, where the network is tested in the end
    Args:
        model: our pytorch model
        test_loader: loader with the validation data
        device: current device (cpu or gpu)
    """
    # testing the model
    model_s.eval()
    model_b.eval()
    test_acc_aligned_s, test_acc_conflicting_s, test_acc_s = evaluate(model_s, test_loader, device)
    test_acc_aligned_b, test_acc_conflicting_b, test_acc_b = evaluate(model_b, test_loader, device)

    wandb.log({"acc_s_test": test_acc_s, "acc_s_test_align": test_acc_aligned_s, "acc_s_test_conflict": test_acc_conflicting_s, "epoch": epochs})
    wandb.log({"acc_b_test": test_acc_b, "acc_b_test_align": test_acc_aligned_b, "acc_b_test_conflict": test_acc_conflicting_b, "epoch": epochs})
    print("test accuracy of unbiased model {:8.3f}".format(test_acc_s))
    print("test accuracy of biased model {:8.3f}".format(test_acc_b))
    return test_acc_s, test_acc_b
    

if __name__ == "__main__":
    main()