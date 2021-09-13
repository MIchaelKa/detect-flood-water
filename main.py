import random

from train import train_model
from utils import get_device, seed_everything

from data import get_train_path_metadata, get_dataset
from data import get_train_metadata, get_datasets

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loss import XEDiceLoss

#
# utils
#

def get_optimizer(name, parameters, lr, weight_decay):
    
    if name == 'Adam':
        half_precision = False
        eps = 1e-4 if half_precision else 1e-08
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps
        )
    elif name == 'SGD':
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )
    else:
        print("[error]: Unsupported optimizer.")
    
    return optimizer

#
# scheduler
#

def test_scheduler(learning_rate, max_iter, scheduler_params):
    lr_history = []

    params = (torch.tensor([1,2,3]) for t in range(2))
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    scheduler = get_scheduler(optimizer, max_iter, scheduler_params)

    for epoch in range(max_iter):
        optimizer.step()
        lr_history.append(scheduler.get_last_lr())
        scheduler.step()

    return lr_history

def get_scheduler(optimizer, max_iter, scheduler_params):
    name = scheduler_params['name']

    if name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_iter,
            eta_min=1e-6,
            last_epoch=-1
        )
    elif name == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=766*2,
            T_mult=2,
            eta_min=1e-6,
            last_epoch=-1
        )
    elif name == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_params['max_lr'],
            total_steps=max_iter,
            anneal_strategy='linear',
            pct_start=scheduler_params['pct_start']
        )
    elif name == 'MultiStepLR':
        milestones = scheduler_params['milestones']
        gamma = scheduler_params['gamma']
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
    elif name == 'StepLR':
        step_size = scheduler_params['step_size']
        gamma = scheduler_params['gamma']
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == 'None':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [max_iter], gamma=1)

    return scheduler

def get_loss(dice_ratio):
    loss = nn.CrossEntropyLoss(ignore_index=255)
    # loss = XEDiceLoss(dice_ratio)
    return loss

def get_model_parameters(model):
    encoder_lr = 3e-5
    decoder_lr = 3e-4

    # parameters = [
    #     {'params': model.encoder.parameters(), 'lr': encoder_lr},
    #     {'params': model.decoder.parameters(), 'lr': decoder_lr}
    # ]

    parameters = model.parameters()

    return parameters

#
# run
#

def run(
    model,
    device,

    path_to_data='./',
    reduce_train=False,
    train_number=0,
    valid_number=0,

    batch_size_train=32,
    batch_size_valid=32,
    max_iter=100,
    unfreeze_iter=0,
    valid_iters=[],

    optimizer_name='Adam',
    learning_rate=3e-4,
    weight_decay=1e-3,
    dice_ratio=0,

    scheduler_params=None,

    save_model=False,
    model_save_name='model',
    
    verbose=True
):

    print('[run]')

    train_dataset, valid_dataset = get_dataset(
        path_to_data, reduce_train, train_number, valid_number
    )

    # TODO: setup
    # num_workers=0,
    # pin_memory=True,
    # batch_size_train
    # batch_size_valid
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        # drop_last=True # TODO: bug with deeplab when batch = 1
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size_valid, shuffle=False)

    print(f'[data] DataLoader size, train: {len(train_loader)}, valid: {len(valid_loader)}')

    num_train_samples = batch_size_train * max_iter
    num_epoch = max_iter / len(train_loader)
    print(f'[data] num_epoch: {num_epoch}, num_train_samples: {num_train_samples}')

    loss = get_loss(dice_ratio)

    optimizer = get_optimizer(optimizer_name, get_model_parameters(model), learning_rate, weight_decay)

    scheduler = get_scheduler(optimizer, max_iter, scheduler_params)

    train_info = train_model(
        fold=0,
        model=model,
        device=device,
        data_loader=train_loader,
        valid_loader=valid_loader,
        criterion=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        max_iter=max_iter,
        unfreeze_iter=unfreeze_iter,
        valid_iters=valid_iters,
        save_model=save_model,
        model_save_name=model_save_name,
        verbose=valid_iters,
        print_every=0
    )

    return train_info

#
# CV
#

import numpy as np
from model import get_model

from utils import format_time
import time

def run_cv(
    device,

    path_to_data='./',
    reduce_train=False,
    train_number=0,
    valid_number=0,

    batch_size_train=32,
    batch_size_valid=32,
    max_iter=100,
    unfreeze_iter=0,
    valid_iters=[],

    optimizer_name='Adam',
    learning_rate=3e-4,
    weight_decay=1e-3,
    dice_ratio=0,

    scheduler_params=None,

    encoder_name='resnet18',
    save_model=False,
    model_save_name='model',
    
    verbose=True
):
    t0 = time.time()
    print('[run_cv]')

    seed = 2021
    seed_everything(seed)

    train_metadata = get_train_metadata(path_to_data)

    flood_ids = train_metadata.flood_id.unique().tolist()
    flood_ids = random.sample(flood_ids, len(flood_ids))

    folds = [
        flood_ids[:3],
        flood_ids[3:6],
        flood_ids[6:9],
        flood_ids[9:],    
    ]

    for f in folds:
        print(f)

    train_infos = []

    for i, fold in enumerate(folds):
        print('')
        print(f'[data] fold: {i}, flood_ids: {fold}')

        train_x, train_y, val_x, val_y = get_train_path_metadata(
            train_metadata,
            fold,
            reduce_train,
            train_number,
            valid_number
        )

        train_dataset, valid_dataset = get_datasets(train_x, train_y, val_x, val_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size_valid, shuffle=False)

        print(f'[data] DataLoader size, train: {len(train_loader)}, valid: {len(valid_loader)}')

        num_train_samples = batch_size_train * max_iter
        num_epoch = max_iter / len(train_loader)
        print(f'[data] num_epoch: {num_epoch}, num_train_samples: {num_train_samples}')

        loss = get_loss(dice_ratio)

        model = get_model(encoder_name).to(device)

        optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate, weight_decay)

        scheduler = get_scheduler(optimizer, max_iter, scheduler_params)

        train_info = train_model(
            fold=i,
            model=model,
            device=device,
            data_loader=train_loader,
            valid_loader=valid_loader,
            criterion=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            max_iter=max_iter,
            unfreeze_iter=unfreeze_iter,
            valid_iters=valid_iters,
            save_model=save_model,
            model_save_name=model_save_name,
            verbose=valid_iters,
            print_every=0
        )

        train_infos.append(train_info)

    cv_scores = np.array([i['best_score'] for i in train_infos])

    print('')
    print(f'[run_cv] results: {cv_scores}')
    print(f'[run_cv] mean: {cv_scores.mean()}, std: {cv_scores.std()}')
    print(f'[run_cv] finished for: {format_time(time.time() - t0)}')

    return train_infos


        