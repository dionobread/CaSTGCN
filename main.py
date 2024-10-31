# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:01:56 2024

@author: Gang Zhang
"""
import os
import random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from Network import Model
from Dataset import CascadeDataset

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# set up device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


def setup_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed_all(seed) # GPU
    os.environ['PYTHONHASHSEED'] = str(seed) 


setup_seed(2024)


def get_max_len(dataset):
    max_len = 0
    for data in tqdm(dataset):
        if len(data.x) > max_len:
            max_len = len(data.x)
            
    return max_len


def msle(y_true, y_pred):
    N = len(y_true)
    sum_error = 0
    for i in range(len(y_true)):
        error = torch.pow((torch.log2(y_pred[i] + 1) - torch.log2(y_true[i] + 1)), 2)
        sum_error += error
    
    res_msle = (1/N) * sum_error
    
    return res_msle


def trainer(model, train_batches, val_batches, optimizer, schedule):
    epoch_loss = []
    val_loss = []
    for i in range(epochs):
        batch_loss = []
        pbar = tqdm(train_batches)
        model.train()
        for batch in pbar:
            batch.to(device)
            optimizer.zero_grad()

            # forward
            pred = model(batch)
            y = batch.y.to(device)

            # compute loss
            loss = msle(y, pred)
            loss_item = loss.item()
            pbar.set_description(f"Epoch{i + 1}, current batch loss: {loss_item:>4f}")

            # backward
            loss.backward()
            optimizer.step()

            batch_loss.append(loss_item)
            

        # validate
        val_avg_loss = tester(model, val_batches)
        schedule.step()
        val_loss.append(val_avg_loss)
        
        batch_avg_loss = sum(batch_loss) / len(batch_loss)
        print(f"Epoch{i + 1} finished, train_loss: {batch_avg_loss:>4f}, val_loss: {val_avg_loss:>4f}\n")
        epoch_loss.append(batch_avg_loss)
        
        
    return epoch_loss, val_loss


def tester(model, test_batches):
    model.eval()
    test_loss_list = []
    with torch.no_grad():
        for batch in test_batches:
            batch.to(device)
            pred = model(batch)
            y = batch.y.to(device)
            
            test_loss = msle(y, pred)
            test_loss_list.append(test_loss.item())
            
    test_avg_loss = sum(test_loss_list) / len(test_loss_list)
    
    return test_avg_loss
    
    
if __name__ == "__main__":
    
    root = "data/weibo/weibo_9"
    dataset = CascadeDataset(root)
    
    print("finding max length...")
    max_len = get_max_len(dataset)
    
    # split the dataset
    N = len(dataset)
    train_start, valid_start, test_start = 0, int(N * 0.7), int(N * (0.8))
    train_data = dataset[train_start:valid_start]
    val_data = dataset[valid_start:test_start]
    test_data = dataset[test_start:N]
    
    batch_size = 64
    
    # create batches
    train_batches = DataLoader(train_data, batch_size=batch_size)
    test_batches = DataLoader(test_data, batch_size=batch_size)
    val_batches = DataLoader(val_data, batch_size=batch_size)
    
    # set up training paramters
    epochs = 20
    learning_rate = 0.0003
    weight_decay = .001
    momentum = 0.9
    
    print("training...")
    model = Model(4, 100, max_len).to(device)
    
    optimizer = torch.optim.SGD(
        model.parameters(), learning_rate, 
        weight_decay=weight_decay, momentum=momentum
    )
    
    schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12], gamma=0.5)
    
    # train
    epoch_loss, val_loss = trainer(model, train_batches, val_batches, optimizer, schedule)

    # test
    test_avg_loss = tester(model, test_batches)
    print(f"average test loss: {test_avg_loss:>4f} \n")
    
    # plot loss curve
    plt.plot(epoch_loss, label='Train loss')
    plt.plot(val_loss, label='Val loss')
    
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.grid(alpha=0.8)
    plt.legend()
    
    plt.savefig("./plots/weibo_9_loss.png", dpi=200)
    plt.close()
    
