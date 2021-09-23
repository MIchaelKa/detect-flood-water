import os
import pandas as pd

from dataset import FloodDataset
from pathlib import Path
from pandas_path import path
from transform import get_train_transform

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

def row_to_path(row, feature):
    chip_id = row.chip_id
    return f'/content/content/train_features/{chip_id}_{feature}.tif'

def get_train_metadata(path_to_data):
    train_metadata = pd.read_csv(
        os.path.join(path_to_data, 'flood-training-metadata.csv'),
        parse_dates=["scene_start"]
    )

    DATA_PATH = Path(path_to_data)

    train_metadata["feature_path"] = (
        str(DATA_PATH / "content/train_features")
        / train_metadata.image_id.path.with_suffix(".tif").path
    )

    train_metadata["label_path"] = (
        str(DATA_PATH / "train_labels")
        / train_metadata.chip_id.path.with_suffix(".tif").path
    )

    return train_metadata


def get_train_path_metadata(
    train_metadata,
    flood_ids,
    reduce_train,
    train_number,
    valid_number
):
    valid_df = train_metadata[train_metadata.flood_id.isin(flood_ids)]
    train_df = train_metadata[~train_metadata.flood_id.isin(flood_ids)]

    features = [
        'nasadem',
        'jrc-gsw-extent',
        'jrc-gsw-occurrence',
        'jrc-gsw-recurrence',
        'jrc-gsw-seasonality',
        'jrc-gsw-transitions',
        'jrc-gsw-change',
    ]

    # Separate features from labels
    val_x = get_paths_by_chip(valid_df)
    val_y = valid_df[["chip_id", "label_path"]].drop_duplicates().reset_index(drop=True)

    train_x = get_paths_by_chip(train_df)
    train_y = train_df[["chip_id", "label_path"]].drop_duplicates().reset_index(drop=True)

    for feature in features:
        val_x[f'{feature}_path'] = val_x.apply(lambda row: row_to_path(row, feature), axis=1)
        train_x[f'{feature}_path'] = train_x.apply(lambda row: row_to_path(row, feature), axis=1)

    valid_ratio = len(val_x) / (len(val_x) + len(train_x)) * 100
    print(f'[data] Dataset size, train: {len(train_x)}, valid: {len(val_x)}, ratio: {valid_ratio}')

    if reduce_train:
        train_x = train_x.head(train_number)
        train_y = train_y.head(train_number)

        val_x = val_x.head(valid_number)
        val_y = val_y.head(valid_number)
        print(f'[data] Reduced dataset size, train: {len(train_x)}, valid: {len(val_x)}')

    return train_x, train_y, val_x, val_y


def get_data_by_flood_id(path_to_data, flood_ids):
    train_metadata = get_train_metadata(path_to_data)

    # data = train_metadata[train_metadata.flood_id==flood_id]
    data = train_metadata[train_metadata.flood_id.isin(flood_ids)]

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

    train_metadata = get_train_metadata(path_to_data)

    # exclude_flood_ids = ['hxu', 'coz']
    # exclude_flood_ids = ['awc', 'ayt', 'coz', 'hbe', 'hxu', 'jja', 'kuo', 'pxs', 'qus', 'qxb', 'tht', 'tnp', 'wvy']
    # exclude_flood_ids = ['coz']
    # train_metadata = train_metadata[~train_metadata.flood_id.isin(exclude_flood_ids)]
    # print(f'[data] exclude_flood_ids: {exclude_flood_ids}')

    # include_flood_ids = ['hxu', 'coz', 'awc']
    # train_metadata = train_metadata[train_metadata.flood_id.isin(include_flood_ids)]
    # print(f'[data] include_flood_ids: {include_flood_ids}')

    # flood_ids = train_metadata.flood_id.unique().tolist()
    # val_flood_ids = random.sample(flood_ids, 3)
    # val_flood_ids = ['kuo', 'tht', 'qus'] # V1
    # val_flood_ids = ['qus', 'hxu', 'pxs'] # V2
    # val_flood_ids = ['jja', 'hbe', 'wvy'] # V3
    # val_flood_ids = ['qxb', 'pxs'] # V4
    val_flood_ids = ['coz', 'hxu', 'pxs'] # V5
    # val_flood_ids = ['coz', 'hxu'] # V6

    # val_flood_ids = ['coz'] # V6


    print(f'[data] flood_ids: {val_flood_ids}')

    return get_train_path_metadata(
        train_metadata,
        val_flood_ids,
        reduce_train,
        train_number,
        valid_number
    )

#
# preprocessing
#

import albumentations as A
from segmentation_models_pytorch.encoders import get_preprocessing_params
import functools
import numpy as np

def preprocess_input(x, mean=None, std=None, **kwargs):

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x

def get_preprocessing_fn(encoder_name, pretrained="imagenet"):
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)

    print('\n[data] preprocessing:')
    print(params)
    print('')

    params['mean'] = params['mean'][:2]
    params['std'] = params['std'][:2]
    return functools.partial(preprocess_input, **params)

#
# get_dataset
#

def get_datasets(
    train_x,
    train_y,
    val_x,
    val_y,
    crop_size,
    encoder_name,
    preprocess_input):
    
    train_transform = get_train_transform(crop_size)
  
    print('\n[data] train_transform:')
    print(train_transform)
    print('')

    if preprocess_input:
        preprocessing_fn = get_preprocessing_fn(encoder_name, pretrained='imagenet')
        preprocessing = A.Lambda(image=preprocessing_fn)
    else:
        preprocessing = None

    features = [
        'vv',
        'vh',
        'nasadem',
        # 'jrc-gsw-extent',
        # 'jrc-gsw-occurrence',
        # 'jrc-gsw-recurrence',
        # 'jrc-gsw-seasonality',
        # 'jrc-gsw-transitions',
        # 'jrc-gsw-change',
    ]

    print(f'\n[data] features: {features}')

    train_dataset = FloodDataset(features, train_x, train_y, transforms=train_transform, preprocessing=preprocessing)
    valid_dataset = FloodDataset(features, val_x, val_y, transforms=None, preprocessing=preprocessing)

    return train_dataset, valid_dataset

def get_dataset(
    path_to_data,
    reduce_train,
    train_number,
    valid_number,
    crop_size,
    encoder_name,
    preprocess_input
    ):

    train_x, train_y, val_x, val_y = prepare_data(path_to_data, reduce_train, train_number, valid_number)
    return get_datasets(train_x, train_y, val_x, val_y, crop_size, encoder_name, preprocess_input)