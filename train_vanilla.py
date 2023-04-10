import os
import yaml
import argparse
import wandb
import time
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader
import torch.optim as optim


from data.util import get_dataset, IdxDataset
from module.util import get_model
from util import set_seed, get_optimizer, evaluate



def main():

    # configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # manual overwriting of configuration for scripts
    #   initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None, help = "Name of experiment")
    parser.add_argument("--bias_conflicting_perc", default=None, type=float, help = "Percentage of bias conflicting samples in dataset")
    parser.add_argument("--severity", default=None, type=int, help = "Severity of bias")
    parser.add_argument("--dataset", default=None, help = "Choice of dataset")
    parser.add_argument("--model_tag", default=None, help = "Choice of model")
    parser.add_argument("--q", default=None, type=float, help = "q for GCE loss")
    parser.add_argument("--random_state", default=None, type=int, help="Random state for seed")
    parser.add_argument("--results_filename", default=None, help="Name of file to store results")
    parser.add_argument("--epochs", default=None, type=int, help="Number of training epochs")
    args = parser.parse_args()
    # Replace all specified arguments
    updateable = [config["name"],config["data"]["bias_conflicting_perc"],config["data"]["severity"],config["data"]["dataset"],config["model"]["tag"],config["loss"]["GCE_q"],config["random_state"],config["results_filename"],config["train"]["epochs"]]
    values = []
    for i,v in enumerate(vars(args).values()):
        if v != None:
             values.append(v)
             print("Overwriting configuration")
        else: values.append(updateable[i])
    config["name"],config["data"]["bias_conflicting_perc"],config["data"]["severity"],config["data"]["dataset"],config["model"]["tag"],config["loss"]["GCE_q"],config["random_state"],config["results_filename"],config["train"]["epochs"] = values

    # configuration sanity check
    if not (
        (config["data"]["dataset"] == "colored_mnist" and config["model"]["tag"] == "MLP") or
        (config["data"]["dataset"] == "colored_mnist" and config["model"]["tag"] == "MLP_VAE") or
        (config["data"]["dataset"] == "cifar10_type0" and config["model"]["tag"] == "ResNet20") or
        (config["data"]["dataset"] == "cifar10_type1" and config["model"]["tag"] == "ResNet20")):
        print("Are you sure you want to use the dataset "+config["data"]["dataset"]+" with the model "+ config["model"]["tag"]+"?")

    # define variables from config
    batch_size = config["train"]["batch_size"]
    epochs = config["train"]["epochs"]
    random_state = config["random_state"]

    # wandb support
    mode = "online" if config['wandb_logging'] else "disabled"
    wandb.init(
        project="Interpretable Debiasing", 
        entity="interpretable-debiasing", 
        config=config, 
        mode=mode
    )
    

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


    train_dataset = IdxDataset(train_dataset)
    test_dataset = IdxDataset(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        #num_workers=16,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        #num_workers=16,
        pin_memory=True,
    )

    # define model
    model = get_model(config).to(device)

    # define optimizer
    optimizer = get_optimizer(model, config)

    # define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # training & validation & test
    for epoch in range(epochs):
        train(model, train_loader, optimizer, criterion, epoch, epochs, device, config)
        #validate()
    test_acc = test(model, test_loader, device)
    
    with open(config["results_filename"]+'.txt', 'a') as f:
        f.writelines((['{} vanilla: {:8.4f}\n'.format(config["name"], test_acc)]))
    timestamp = time.strftime(' %d-%b-%Y_%H:%M', time.localtime())
    os.makedirs("./saved_models/vanilla/", exist_ok=True)
    torch.save(model.state_dict(), "./saved_models/vanilla/" + config["name"] + timestamp + ".pth")
    wandb.finish(quiet=True)


def train(
    model,
    train_loader,
    optimizer,
    criterion,
    epoch,
    epochs,
    device,
    config
):
    """Main training loop, where the network is trained
    Args:
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

    total_acc, total_count = 0, 0
    # training loop
    model.train()
    for idx, (data_index, data, attr) in enumerate(train_loader):
        data, attr = data.to(device), attr.to(device)
        label = attr[:, 0]
        # bias = attr[:, 1]

        optimizer.zero_grad()
        mean, logvar = model.encoder(data)
        logit = model.predict(mean)

        loss = criterion(logit,label)
        loss.backward()
        optimizer.step()

        corr = (logit.argmax(1) == label).sum().item()
        batch_len = label.size(0)

        total_acc += corr
        total_count += batch_len
        
        train_loader.set_postfix(loss=loss.item(), acc= corr / batch_len)
        wandb.log({"train_loss": loss})

        
    wandb.log({"train_accuracy": total_acc / total_count, "epoch": epoch})
    print(
        "| epoch {:3d} | training accuracy {:8.3f}".format(
            epoch, total_acc / total_count
        )
    )


def test(model, test_loader, device):
    """Main test loop, where the network is tested in the end
    Args:
        model: our pytorch model
        test_loader: loader with the validation data
        device: current device (cpu or gpu)
    """
    # testing the model
    model.eval()
    test_acc_aligned, test_acc_conflicting, test_acc = evaluate(model, test_loader, device)
    wandb.log({"conflicting_test_accuracy_vanilla": test_acc_conflicting})
    wandb.log({"aligned_test_accuracy_vanilla": test_acc_aligned})   
    wandb.log({"test_accuracy_vanilla": test_acc})
    print("test accuracy {:8.3f}".format(test_acc))
    return test_acc
    

   
if __name__ == "__main__":
   main()