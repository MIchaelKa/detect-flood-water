import os
import random
import pandas as pd

from train import train_model
from utils import get_device, seed_everything

from dataset import FloodDataset
from pathlib import Path
from pandas_path import path
from transform import get_train_transform

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

#
# data
#

# Helper function for pivoting out paths by chip
def get_paths_by_chip(image_level_df):
    """
    Returns a chip-level dataframe with pivoted columns
    for vv_path and vh_path.

    Args:
        image_level_df (pd.DataFrame): image-level dataframe

    Returns:
        chip_level_df (pd.DataFrame): chip-level dataframe
    """
    paths = []
    for chip, group in image_level_df.groupby("chip_id"):
        vv_path = group[group.polarization == "vv"]["feature_path"].values[0]
        vh_path = group[group.polarization == "vh"]["feature_path"].values[0]
        flood_id = group["flood_id"].values[0]
        paths.append([chip, vv_path, vh_path, flood_id])
    return pd.DataFrame(paths, columns=["chip_id", "vv_path", "vh_path", "flood_id"])

def get_data_by_flood_id(path_to_data, flood_id):
    train_metadata = pd.read_csv(
        os.path.join(path_to_data, 'flood-training-metadata.csv'),
        parse_dates=["scene_start"]
    )

    DATA_PATH = Path(path_to_data)

    train_metadata["feature_path"] = (
        str(DATA_PATH / "train_features")
        / train_metadata.image_id.path.with_suffix(".tif").path
    )

    train_metadata["label_path"] = (
        str(DATA_PATH / "train_labels")
        / train_metadata.chip_id.path.with_suffix(".tif").path
    )

    data = train_metadata[train_metadata.flood_id==flood_id]

    x = get_paths_by_chip(data)
    y = data[["chip_id", "label_path"]].drop_duplicates().reset_index(drop=True)

    print(f'[data] Dataset size: {len(x)}')

    return x, y


def prepare_data(
    path_to_data,
    reduce_train,
    train_number,
    valid_number
    ):

    train_metadata = pd.read_csv(
        os.path.join(path_to_data, 'flood-training-metadata.csv'),
        parse_dates=["scene_start"]
    )

    DATA_PATH = Path(path_to_data)

    train_metadata["feature_path"] = (
        str(DATA_PATH / "train_features")
        / train_metadata.image_id.path.with_suffix(".tif").path
    )

    train_metadata["label_path"] = (
        str(DATA_PATH / "train_labels")
        / train_metadata.chip_id.path.with_suffix(".tif").path
    )

    # flood_ids = train_metadata.flood_id.unique().tolist()
    # val_flood_ids = random.sample(flood_ids, 3)
    # val_flood_ids = ['kuo', 'tht', 'qus'] # V1
    val_flood_ids = ['qus', 'hxu', 'pxs'] # V2
    # val_flood_ids = ['jja', 'hbe', 'wvy'] # V3
    print(f'[data] flood_ids: {val_flood_ids}')

    valid_df = train_metadata[train_metadata.flood_id.isin(val_flood_ids)]
    train_df = train_metadata[~train_metadata.flood_id.isin(val_flood_ids)]

    # Separate features from labels
    val_x = get_paths_by_chip(valid_df)
    val_y = valid_df[["chip_id", "label_path"]].drop_duplicates().reset_index(drop=True)

    train_x = get_paths_by_chip(train_df)
    train_y = train_df[["chip_id", "label_path"]].drop_duplicates().reset_index(drop=True)

    valid_ratio = len(val_x) / (len(val_x) + len(train_x)) * 100
    print(f'[data] Dataset size, train: {len(train_x)}, valid: {len(val_x)}, ratio: {valid_ratio}')

    if reduce_train:
        train_x = train_x.head(train_number)
        train_y = train_y.head(train_number)

        val_x = val_x.head(valid_number)
        val_y = val_y.head(valid_number)
        print(f'[data] Reduced dataset size, train: {len(train_x)}, valid: {len(val_x)}')

    return train_x, train_y, val_x, val_y


def get_dataset(
    path_to_data,
    reduce_train,
    train_number,
    valid_number
    ):

    train_x, train_y, val_x, val_y = prepare_data(path_to_data, reduce_train, train_number, valid_number)  

    # TODO: play with it!
    crop_size = 256
    train_transform = get_train_transform(crop_size)
  
    # print('\n[data] train_transform:')
    # print(train_transform)
    # print('')

    train_dataset = FloodDataset(train_x, train_y, transforms=train_transform)
    valid_dataset = FloodDataset(val_x, val_y, transforms=None)

    return train_dataset, valid_dataset

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size_valid, shuffle=False)

    print(f'[data] DataLoader size, train: {len(train_loader)}, valid: {len(valid_loader)}')

    num_train_samples = batch_size_train * max_iter
    num_epoch = max_iter / len(train_loader)
    print(f'[data] num_epoch: {num_epoch}, num_train_samples: {num_train_samples}')

    loss = nn.CrossEntropyLoss(ignore_index=255)
    # loss = XEDiceLoss(dice_ratio=0)

    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate, weight_decay)

    scheduler = get_scheduler(optimizer, max_iter, scheduler_params)

    train_info = train_model(
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
        # print_every=2
    )

    train_info['valid_loader'] = valid_loader
    train_info['valid_dataset'] = valid_dataset

    return train_info
