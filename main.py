import os
import random
import pandas as pd

from train import train_model
from utils import get_device, seed_everything

from dataset import FloodDataset
from pathlib import Path
from pandas_path import path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
        paths.append([chip, vv_path, vh_path])
    return pd.DataFrame(paths, columns=["chip_id", "vv_path", "vh_path"])

def get_dataset(
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

    flood_ids = train_metadata.flood_id.unique().tolist()
    val_flood_ids = random.sample(flood_ids, 3)
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

    train_dataset = FloodDataset(train_x, train_y, transforms=None)
    valid_dataset = FloodDataset(val_x, val_y, transforms=None)

    return train_dataset, valid_dataset

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
    valid_iters=[],

    optimizer_name='Adam',
    learning_rate=3e-4,
    weight_decay=1e-3,

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

    loss = nn.CrossEntropyLoss(ignore_index=255)

    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate, weight_decay)

    train_info = train_model(
        model=model,
        device=device,
        data_loader=train_loader,
        valid_loader=valid_loader,
        criterion=loss,
        optimizer=optimizer,
        max_iter=max_iter,
        valid_iters=valid_iters,
        verbose=valid_iters,
        # print_every=2
    )

    return train_info