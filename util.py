import os
import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchvision.transforms.functional import normalize




def set_seed(seed):
    """
    Set all random seeds
    Args:
        seed (int): integer for reproducible experiments
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_optimizer(model, config):
    """Resolve the optimizer according to the configs
    Args:
        model: model on which the optimizer is applied on
        config: configuration dict
    returns:
        optimizer
    """
    if config["optimizer"]["name"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["optimizer"]["lr"],
            momentum=config["optimizer"]["momentum"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
    elif config["optimizer"]["name"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["optimizer"]["lr"],
            betas=config["optimizer"]["betas"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
    else: raise NotImplementedError("Optimizer not implemented.")
    return optimizer

        
class GeneralizedCELoss(nn.Module):

    def __init__(self, config):
        super(GeneralizedCELoss, self).__init__()
        self.q = config["loss"]["GCE_q"]
             
    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError("GCE_p")
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q) # Do we really need *self.q? I think like now is correct.
        if np.isnan(Yg.mean().item()):
            raise NameError("GCE_Yg")

        # note that we don't return the average but the loss for each datum separately
        loss = F.cross_entropy(logits, targets, reduction="none") * loss_weight 

        return loss


    # evaluation code for testset
def evaluate(model, test_loader, device):
    model.eval()
    test_loader = tqdm(test_loader, position=0, leave=False)
    total_correct_aligned, total_num_aligned = 0, 0
    total_correct_conflicting, total_num_conflicting = 0, 0
    for idx, (data_index, data, attr) in enumerate(test_loader):
        label = attr[:, 0]
        bias = attr[:, 1]
        data = data.to(device)
        label = label.to(device)
        bias = bias.to(device)

        with torch.no_grad():
            # For evaluation take mean directly to not unnecessarily introduce variance
            parameters = model.encoder(data)
            assert len(parameters) == 2 # No new outputs of encoders
            pred = model.predict(parameters[0]).argmax(1)
            correct = (pred == label).long()
            aligned = (label == bias).long()
            total_correct_aligned += correct[aligned==True].sum()
            total_correct_conflicting += correct[aligned==False].sum()
            total_num_aligned += correct[aligned==True].size(0)
            total_num_conflicting += correct[aligned==False].size(0)


    acc_aligned = total_correct_aligned/float(total_num_aligned)
    acc_conflicting = total_correct_conflicting/float(total_num_conflicting)
    acc = (total_correct_aligned+total_correct_conflicting)/(float(total_num_aligned)+float(total_num_conflicting))
    model.train()

    return acc_aligned, acc_conflicting, acc


def evaluate_batch(logit, attr, loss):
    label = attr[:, 0]
    bias = attr[:, 1]
    
    pred = logit.data.argmax(1)
    correct = (pred == label).long()
    aligned = (label == bias).long()

    aligned_len = aligned.sum()
    conflicting_len = (1-aligned).sum()
    batch_len = label.size(0)
    assert(batch_len == aligned_len + conflicting_len)

    corr_aligned = correct[aligned==True].sum()
    corr_conflicting = correct[aligned==False].sum() 
    corr = correct.sum()
    assert(corr == corr_aligned + corr_conflicting)

    loss_aligned = loss[aligned==True].mean()
    loss_conflicting = loss[aligned==False].mean()
    loss = loss.mean()

    return corr_aligned, corr_conflicting, corr, loss_aligned, loss_conflicting, loss, aligned_len, conflicting_len, batch_len


def save_img(model_s, model_b, data_loader, config, device):
    # Logging image
    set_seed(config["random_state"])
    model_s.eval()
    model_b.eval()
    sample1, sample2 = random.sample(list(data_loader), 2)
    data = torch.stack((sample1[1][0],sample2[1][0]))
    data = data.to(device)

    z_s, logits_s, mean_s, logvar_s = model_s(data)
    z_b, logits_b, mean_b, logvar_b = model_b(data)
    mean = torch.cat((mean_s, mean_b), dim=1)
    swap1 = torch.cat((mean_s[0],mean_b[1]))
    swap2 = torch.cat((mean_s[1],mean_b[0]))
    x_reconstructed = model_s.reconstruct(mean)
    mean[0] = swap1
    mean[1] = swap2
    swap_reconstr = model_s.reconstruct(mean)

    if config["data"]["dataset"] == "colored_mnist":
        data = data.view(2,3,28,28)
        x_reconstructed = x_reconstructed.view(2,3,28,28)
        swap_reconstr = swap_reconstr.view(2,3,28,28)
    elif (config["data"]["dataset"] == "cifar10_type0" or config["data"]["dataset"] == "cifar10_type1"): # Backtransform preprocessing standardization for CE
        data = normalize(data,-np.divide([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]), np.divide([1,1,1],[0.2023, 0.1994, 0.2010]))
 

    import torchvision
    save_img = torchvision.utils.make_grid([data[0],x_reconstructed[0],swap_reconstr[0],
                                            data[1],x_reconstructed[1],swap_reconstr[1]],nrow=3)
    save_img = wandb.Image(save_img, caption="Left: Original image, Middle: Reconstructed image, Right: Keeping signal, swapping bias")
    wandb.log({"Visualization": save_img})

def save_img_adv(model_s, model_b, data_loader, epoch, config, device, training=False):
    # Logging image
    model_s.eval()
    model_b.eval()

    rand_batches = random.sample(list(data_loader), 5)
    
    if training == True: 
        data_batches = [item[2] for item in rand_batches] # 1 higher index because we also have subsetindex in trainset
        attr = [item[3] for item in rand_batches]
        data = torch.stack([item[0] for item in data_batches])
        label = torch.stack([item[0,0] for item in attr])
    else: 
        data_batches = [item[1] for item in rand_batches]
        attr = [item[2] for item in rand_batches]
        data_unpacked = list()
        attr_unpacked = list()
        for index, item in enumerate(attr):
            idx = torch.where(item[:,0] == item[:,1])[0][0]
            data_unpacked.append(data_batches[index][idx])
            attr_unpacked.append(item[idx])
        data = torch.stack(data_unpacked)
        label = torch.stack(attr_unpacked)[:,0]
    data = data.to(device)
    label = label.to(device)
    assert data.shape[0:2] ==torch.Size([5, 3])
    z_s, logits_s, mean_s, logvar_s = model_s(data)
    z_b, logits_b, mean_b, logvar_b = model_b(data)
    attack = DeepFool(model_b.classifier,device,steps=10,overshoot=config["perturb"]["overshoot"])
    mean_b_adv, label_adv = attack.forward(mean_b, label)
    mean = torch.cat((mean_s, mean_b), dim=1)
    mean_adv = torch.cat((mean_s, mean_b_adv), dim=1)
    x_reconstructed = model_s.reconstruct(mean)
    x_adv_reconstr = model_s.reconstruct(mean_adv)

    if config["data"]["dataset"] == "colored_mnist":
        data = data.view(5,3,28,28)
        x_reconstructed = x_reconstructed.view(5,3,28,28)
        x_adv_reconstr = x_adv_reconstr.view(5,3,28,28)
    elif (config["data"]["dataset"] == "cifar10_type0" or config["data"]["dataset"] == "cifar10_type1"): # Backtransform preprocessing standardization for CE
        data = normalize(data,-np.divide([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]), np.divide([1,1,1],[0.2023, 0.1994, 0.2010]))
 

    import torchvision
    imgs = torch.cat((data, x_reconstructed,x_adv_reconstr))
    save_img = torchvision.utils.make_grid(imgs,nrow=5)
    save_img = wandb.Image(save_img, caption="Top: Original image, Middle: Reconstructed image, Bottom: Reconstructed adv. perturbation")
    if training == True: 
        wandb.log({"Adversarial Visualization Training": save_img, "epoch": epoch})
    else: 
        wandb.log({"Adversarial Visualization": save_img, "epoch": epoch})
    model_s.train()
    model_b.train()

 ## DeepFool is adapted from https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/deepfool.html
class DeepFool:
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 50)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)

    Adaptation Note: This algorithm is designed for the purpose of this work.
    Therefore it does not work with bounded images but unbounded latent dimensions. 
    We still call the input "image" for consistency with the original algorithm.
    Additionally the algorithm is optimized by approximating the closest hyperplane by the second highest predicted class.
    This is in order to reduce computational complexity 50fold as it allows parallelizing. 
    In my tests the approximation gave the optimal(Oracle being original DeepFool) solution in 4/5 of cases.
    

    Examples::
        >>> attack = DeepFool(model, steps=50, overshoot=0.02)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, device, steps=50, overshoot=0.02):
        self.model = model
        self.steps = steps
        self.overshoot = overshoot
        self.device = device
        self.num_classes = model.num_classes

    def forward(self, images, labels, target_true_label=False):
        adv_images = images.clone().detach().to(self.device)
        adv_images.requires_grad = True
        labels = labels.clone().detach().to(self.device)
        batch_size = len(adv_images)
        correct = torch.tensor([True]*batch_size).to(self.device)
        if target_true_label:
            target_label = labels
        else:
            target_labels = torch.ones([batch_size,self.num_classes])/(self.num_classes-1)
            target_labels = target_labels.to(self.device)
            for i in range(self.num_classes):
                target_labels[labels == i,i] = 0
            target_label = torch.multinomial(target_labels,1).squeeze(-1)
        curr_steps = 0        

        # Note that with this implementation it's possible that the target label switches between iterations
        while (True in correct) and (curr_steps < self.steps):
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            logits = self.model(adv_images[correct]) # Forward pass only for correct classifications
            values, predicted = torch.topk(logits, 2, dim=1) # Predicted label            
            #target_label[correct] = predicted[:,0]
            correct_new = (predicted[:,0]!=target_label[correct])
            #correct_new = (predicted[:,0]==labels[correct])
            # Some indexing to backprop only wrt correctly classified labels
            #diff = values[:,1] - logits.gather(1,labels[correct].unsqueeze(1)).squeeze(-1) # second highest label as target
            diff = logits.gather(1,target_label[correct].unsqueeze(1)).squeeze(-1)  - values[:,0] # target label as target

            diff[~correct_new] = 0
            diff_backprop = diff.sum() # "Trick" to backprop wrt all inputs: Summing individual differences
            diff_backprop.backward()
            delta = (torch.abs(diff[correct_new])/(torch.norm(adv_images.grad[correct][correct_new], p=2,dim=tuple(range(adv_images.ndim)[1:]))**2+1e-8)).view(-1,*(1,)*(adv_images.dim()-1)) * adv_images.grad[correct][correct_new]
            assert not torch.isnan(delta).any()
            correct[correct.clone()] = correct_new
            with torch.no_grad():
                adv_images[correct] = (adv_images[correct] + (1+self.overshoot)*delta).detach()
            curr_steps += 1

        return adv_images, target_label


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=3, verbose=False, delta=0, path_s = 'checkpoint.pt', path_b = None, trace_func=print, saveEveryEpoch=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path_s = path_s
        self.path_b = path_b
        self.trace_func = trace_func
        self.saveEveryEpoch = saveEveryEpoch

    def __call__(self, val_loss, model_s, model_b, epoch):
        score = -val_loss

        if self.saveEveryEpoch:
            path_s = self.path_s[:-3] + "_epoch_" + str(epoch) + ".pt"
            path_b = self.path_b[:-3] + "_epoch_" + str(epoch) + ".pt"

            self.save_checkpoint(val_loss, model_s, model_b, path_s, path_b)


        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_s, model_b, self.path_s, self.path_b)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.trace_func(f'Early Stopping biased model at epoch {self.counter}')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_s, model_b, self.path_s, self.path_b)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_s, model_b, path_s, path_b):
        """Saves model when validation loss decreases."""

        if self.verbose:
            self.trace_func(f'Biased val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving models...\n')
        torch.save(model_s.state_dict(), path_s)
        torch.save(model_b.state_dict(), path_b)

        self.val_loss_min = val_loss




def get_reconst_loss(model_s, model_b, data_loader, device, config, mode="train"):
    """Main test loop, where the network is tested in the end
    Args:
        model: our pytorch model
        val_loader: loader with the validation data
        device: current device (cpu or gpu)
    """
    # testing the model
    model_s.eval()
    model_b.eval()
    data_loader = tqdm(data_loader, position=0, leave=False)
    total_loss = 0
    if mode=="train":
        for idx, (subset_idx, full_idx, data, attr) in enumerate(data_loader):
            label = attr[:, 0]
            data = data.to(device)
            label = label.to(device)

            with torch.no_grad():
                # For evaluation take mean directly to not unnecessarily introduce variance
                parameters = model_s.encoder(data)
                parameters_b = model_b.encoder(data)
                assert len(parameters) == 2 # No new outputs of encoders
                x_reconst = model_s.decoder(torch.cat((parameters[0],parameters_b[0]),dim=1))
                if config["data"]["dataset"] == "colored_mnist":
                    reconst_loss = F.binary_cross_entropy(x_reconst, data, reduction='none').sum(dim=(1,2,3))
                    reconst_loss /= x_reconst.view(len(x_reconst),-1).shape[1]
                elif (config["data"]["dataset"] == "cifar10_type0" or config["data"]["dataset"] == "cifar10_type1"): # Backtransform preprocessing standardization for CE
                    data_backtransformed = normalize(data,-np.divide([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]), np.divide([1,1,1],[0.2023, 0.1994, 0.2010]))
                    reconst_loss = F.mse_loss(x_reconst, data_backtransformed, reduction='none').sum(dim=(1,2,3))
                    reconst_loss /= x_reconst.view(len(x_reconst),-1).shape[1]
                elif (config["data"]["dataset"] == "camelyon17_type0" or config["data"]["dataset"] == "camelyon17_type1" or config["data"]["dataset"] == "camelyon17_type2"):
                    reconst_loss = F.mse_loss(x_reconst, data, reduction='none').sum(dim=(1,2,3))
                    reconst_loss /= x_reconst.view(len(x_reconst),-1).shape[1]
                else: raise NotImplementedError("reconst_loss")
                total_loss += reconst_loss.sum()
    elif mode=="test":
        for idx, (full_idx, data, attr) in enumerate(data_loader):
            label = attr[:, 0]
            data = data.to(device)
            label = label.to(device)

            with torch.no_grad():
                # For evaluation take mean directly to not unnecessarily introduce variance
                parameters = model_s.encoder(data)
                parameters_b = model_b.encoder(data)
                assert len(parameters) == 2 # No new outputs of encoders
                x_reconst = model_s.decoder(torch.cat((parameters[0],parameters_b[0]),dim=1))
                if config["data"]["dataset"] == "colored_mnist":
                    reconst_loss = F.binary_cross_entropy(x_reconst, data, reduction='none').sum(dim=(1,2,3))
                    reconst_loss /= x_reconst.view(len(x_reconst),-1).shape[1]
                elif (config["data"]["dataset"] == "cifar10_type0" or config["data"]["dataset"] == "cifar10_type1"): # Backtransform preprocessing standardization for CE
                    data_backtransformed = normalize(data,-np.divide([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]), np.divide([1,1,1],[0.2023, 0.1994, 0.2010]))
                    reconst_loss = F.mse_loss(x_reconst, data_backtransformed, reduction='none').sum(dim=(1,2,3))
                    reconst_loss /= x_reconst.view(len(x_reconst),-1).shape[1]
                elif (config["data"]["dataset"] == "camelyon17_type0" or config["data"]["dataset"] == "camelyon17_type1" or config["data"]["dataset"] == "camelyon17_type2"):
                    reconst_loss = F.mse_loss(x_reconst, data, reduction='none').sum(dim=(1,2,3))
                    reconst_loss /= x_reconst.view(len(x_reconst),-1).shape[1]
                else: raise NotImplementedError("reconst_loss")
                total_loss += reconst_loss.sum()
    total_loss /= len(data_loader.iterable.dataset)
    print("total reconstruction loss of VAE: {:8.6f}".format(total_loss))
    return total_loss


def early_stop(model_s, model_b, val_loader_biased, stopping_criteria, scheduler_s, scheduler_b, epoch, device, config):
    """Main test loop, where the network is tested in the end
    Args:
        model: our pytorch model
        val_loader: loader with the validation data
        device: current device (cpu or gpu)
    """
    # testing the model
    model_b.eval()
    model_s.eval()
    val_loader_biased = tqdm(val_loader_biased, position=0, leave=False)
    total_loss = 0
    total_reconst_loss = 0
    for idx, (subset_idx, full_idx, data, attr) in enumerate(val_loader_biased):
        label = attr[:, 0]
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            # For evaluation take mean directly to not unnecessarily introduce variance
            parameters_b = model_b.encoder(data)
            assert len(parameters_b) == 2 # No new outputs of encoders
            logits_b = model_b.predict(parameters_b[0])
            parameters_s = model_s.encoder(data)
            assert len(parameters_s) == 2 # No new outputs of encoders
            logits_s = model_s.predict(parameters_s[0])
            loss_s = F.cross_entropy(logits_s, label,reduction="none")
            loss_b = F.cross_entropy(logits_b, label,reduction="none")
            mean = torch.cat((parameters_s[0], parameters_b[0]), dim=1)
            logvar = torch.cat((parameters_s[1], parameters_b[1]), dim=1)
            x_reconst = model_s.reconstruct(mean)

            p = F.softmax(logits_b, dim=1)
            if np.isnan(p.mean().item()):
                raise NameError("GCE_p")
            Yg = torch.gather(p, 1, torch.unsqueeze(label, 1))
            # modify gradient of cross entropy
            loss_weight = (Yg.squeeze().detach()) # Do we really need *self.q? I think like now is correct.
            if np.isnan(Yg.mean().item()):
                raise NameError("GCE_Yg")

            # note that we don't return the average but the loss for each datum separately
            loss_b = (loss_b * loss_weight**config["loss"]["GCE_q"])
            loss_s = (loss_s * (1-loss_weight)**config["loss"]["GCE_q"])
            # VAE losses     
            # Compute reconstruction loss and kl divergence for both encoders together
            # Sum over dimensions, average over batch to have loss weighting hyperparameters being independent of batch size
            if (config["data"]["dataset"] == "cifar10_type0" or config["data"]["dataset"] == "cifar10_type1"): # Backtransform preprocessing standardization for CE
                data_backtransformed = normalize(data,-np.divide([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]), np.divide([1,1,1],[0.2023, 0.1994, 0.2010]))
                reconst_loss = F.mse_loss(x_reconst, data_backtransformed, reduction='none').sum(dim=(1,2,3))
            elif config["data"]["dataset"] == "colored_mnist":
                reconst_loss = F.binary_cross_entropy(x_reconst, data, reduction='none').sum(dim=(1,2,3))
            elif (config["data"]["dataset"] == "camelyon17_type0" or config["data"]["dataset"] == "camelyon17_type1"or config["data"]["dataset"] == "camelyon17_type2"):
                reconst_loss = F.mse_loss(x_reconst, data, reduction='none').sum(dim=(1,2,3))
            else: raise NotImplementedError("reconst_loss")
            kl_div = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(),dim=1)

            # Rescaling both VAE losses in order to be invariant to image resolution in hyperparametertuning.
            reconst_loss /= x_reconst.view(len(x_reconst),-1).shape[1]
            kl_div /= x_reconst.view(len(x_reconst),-1).shape[1]

            reconst_loss *=config["loss"]["reconst_weight"]

            total_loss = total_loss + loss_s.sum() + loss_b.sum() + reconst_loss.sum() + kl_div.sum()
            total_reconst_loss += reconst_loss.sum()
    total_loss /= len(val_loader_biased.iterable.dataset)
    total_reconst_loss /= len(val_loader_biased.iterable.dataset)
    # scheduling
    if config["optimizer"]["lr_decay"]:
        scheduler_s.step(total_loss)
        scheduler_b.step(total_loss)
    wandb.log({"loss_biasedval": total_loss, 'reconstruction_loss_biasedval': total_reconst_loss, "epoch": epoch})
    print("total validation loss {:8.3f}".format(total_loss))
    stopping_criteria(total_loss, model_s, model_b, epoch)
    model_s.train()
    model_b.train()
    return stopping_criteria.early_stop


def capture_dataset(data_loader,config):
    imgs = []
    for idx, (subset_idx, full_idx, data, attr) in enumerate(data_loader):

        if (config["data"]["dataset"] == "cifar10_type0" or config["data"]["dataset"] == "cifar10_type1"): # Backtransform preprocessing standardization for CE
            data = normalize(data,-np.divide([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]), np.divide([1,1,1],[0.2023, 0.1994, 0.2010]))
        label = attr[:, 0]
        bias = attr[:,1]
        aligned = (label==bias)
        data_aligned = data[aligned]
        label_aligned = label[aligned]
        for j in range(3):
            for i in np.unique(label_aligned):
                imgs.append(data_aligned[label_aligned==i][j])
        import torchvision
        save_img = torchvision.utils.make_grid(imgs,nrow=len(np.unique(label_aligned)))
        save_img = wandb.Image(save_img)
        wandb.log({"Dataset": save_img})
        break


def bias_visualization(model_s, model_b, data_loader, config, device):
    # Visualizing Bias. 

    model_s.eval()
    model_b.eval()

    rand_batches = random.sample(list(data_loader), 5)

    data_batches = [item[2] for item in rand_batches]
    attr = [item[3] for item in rand_batches]
    data_unpacked = list()
    attr_unpacked = list()
    for index, item in enumerate(attr):
        # Extract 5 bias-conflicting images (w/o using bias label as it's theoretically unknown)
        batch = data_batches[index].to(device)
        item = item.to(device)
        parameters_b = model_b.encoder(batch)
        assert len(parameters_b) == 2 # No new outputs of encoders
        pred_b = model_b.predict(parameters_b[0]).argmax(1)
        correct_b = (pred_b == item[:,0]).long()
        parameters_s = model_s.encoder(batch)
        assert len(parameters_s) == 2 # No new outputs of encoders
        pred_s = model_s.predict(parameters_s[0]).argmax(1)
        correct_s = (pred_s == item[:,0]).long()
        bias_aligned = (correct_s*correct_b).bool()
        data_unpacked.append(data_batches[index][bias_aligned.cpu()][0])
        attr_unpacked.append(item[bias_aligned.cpu()][0])
    data = torch.stack(data_unpacked)
    label = torch.stack(attr_unpacked)[:,0]
    data = data.to(device)
    label = label.to(device)
    assert data.shape[0:2] ==torch.Size([5, 3])
    z_s, logits_s, mean_s, logvar_s = model_s(data)
    z_b, logits_b, mean_b, logvar_b = model_b(data)
    attack = DeepFool(model_b.classifier,device,steps=20,overshoot=config["perturb"]["overshoot"])
    mean_b_adv, label_adv = attack.forward(mean_b, label)
    mean = torch.cat((mean_s, mean_b), dim=1)
    mean_adv = torch.cat((mean_s, mean_b_adv), dim=1)
    x_reconstructed = model_s.reconstruct(mean)
    x_adv_reconstr = model_s.reconstruct(mean_adv)

##### FOR DFA THEN FIND SAMPLES WHERE BIAS PREDICTS LABEL OF DATA#####
    # Create bias-aligned samples by finding samples whose bias dimensions makes biased classifier predict correct label.
    j = 0
    mean_b_swap = torch.zeros_like(mean_b)
    while j<5:
        rand_batch = random.sample(list(data_loader), 1)
        batch_data = rand_batch[0][2].to(device)
        #batch_label = rand_batch[0][2][:,0].to(device)
        parameters_b = model_b.encoder(batch_data)
        assert len(parameters_b) == 2 # No new outputs of encoders
        pred_b = model_b.predict(parameters_b[0]).argmax(1)
        corr_bias = (pred_b == label_adv[j])
        if corr_bias.sum()>0:
            mean_b_swap[j] = parameters_b[0][corr_bias][0] 
            j+=1

    mean_swap = torch.cat((mean_s, mean_b_swap), dim=1)
    x_swap_reconstr = model_s.reconstruct(mean_swap)



    if config["data"]["dataset"] == "colored_mnist":
        data = data.view(5,3,28,28)
        x_reconstructed = x_reconstructed.view(5,3,28,28)
        x_adv_reconstr = x_adv_reconstr.view(5,3,28,28)
        x_swap_reconstr = x_swap_reconstr.view(5,3,28,28)
    elif (config["data"]["dataset"] == "cifar10_type0" or config["data"]["dataset"] == "cifar10_type1"): # Backtransform preprocessing standardization for CE
        data = normalize(data,-np.divide([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]), np.divide([1,1,1],[0.2023, 0.1994, 0.2010]))
 

    import torchvision
    imgs = torch.cat((data, x_reconstructed,x_adv_reconstr))
    save_img = torchvision.utils.make_grid(imgs,nrow=5)
    save_img = wandb.Image(save_img, caption="Top: Original image, Middle: Reconstructed bias-conflicting image, Bottom: Reconstructed bias-aligned image by adv. perturbation")
    wandb.log({"Adversarial Visualization Ours": save_img})


    imgs = torch.cat((data, x_reconstructed,x_swap_reconstr))
    save_img = torchvision.utils.make_grid(imgs,nrow=5)
    save_img = wandb.Image(save_img, caption="Top: Original image, Middle: Reconstructed bias-conflicting image, Bottom: Reconstructed bias-aligned image by swapping")
    wandb.log({"Adversarial Visualization DisEnt": save_img})

    model_s.train()
    model_b.train()





