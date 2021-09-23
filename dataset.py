
import numpy as np
import rasterio

from torch.utils.data import Dataset

class FloodDataset(Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def __init__(self, features, x_paths, y_paths=None, transforms=None, preprocessing=None):
        self.features = features
        self.data = x_paths
        self.label = y_paths
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img = self.data.loc[idx]
        images = []

        for feature in self.features:
            with rasterio.open(img[f'{feature}_path']) as f:
                images.append(f.read(1))

        x_arr = np.stack(images, axis=-1)

        # Min-max normalization
        x_arr_max = np.max(x_arr, axis=(0,1))
        x_arr_min = np.min(x_arr, axis=(0,1))

        x_arr = (x_arr - x_arr_min) / (x_arr_max - x_arr_min)

        # Load label if available - training only
        if self.label is not None:
            label_path = self.label.loc[idx].label_path
            with rasterio.open(label_path) as lp:
                y_arr = lp.read(1)

            # Apply data augmentations, if provided
            if self.transforms:
                t = self.transforms(image=x_arr, mask=y_arr)
                x_arr, y_arr = t['image'], t['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=x_arr, mask=y_arr)
            x_arr, y_arr = sample['image'], sample['mask']


        x_arr = np.transpose(x_arr, [2, 0, 1])

        sample = {
            "chip_id": img.chip_id,
            "chip": x_arr,
            "label": y_arr,
            "flood_id": img.flood_id}

        return sample