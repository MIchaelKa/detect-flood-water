
import numpy as np
import rasterio

from torch.utils.data import Dataset

class FloodDataset(Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def __init__(self, x_paths, y_paths=None, transforms=None):
        self.data = x_paths
        self.label = y_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Loads a 2-channel image from a chip-level dataframe
        img = self.data.loc[idx]
        with rasterio.open(img.vv_path) as vv:
            vv_path = vv.read(1)
        with rasterio.open(img.vh_path) as vh:
            vh_path = vh.read(1)
        x_arr = np.stack([vv_path, vh_path], axis=-1)

        # Min-max normalization
        min_norm = -77
        max_norm = 26
        x_arr = np.clip(x_arr, min_norm, max_norm)
        x_arr = (x_arr - min_norm) / (max_norm - min_norm)

        # Load label if available - training only
        if self.label is not None:
            label_path = self.label.loc[idx].label_path
            with rasterio.open(label_path) as lp:
                y_arr = lp.read(1)

            # Apply data augmentations, if provided
            if self.transforms:
                t = self.transforms(image=x_arr, mask=y_arr)
                x_arr, y_arr = t['image'], t['mask']

        x_arr = np.transpose(x_arr, [2, 0, 1])
        sample = {"chip_id": img.chip_id, "chip": x_arr, "label": y_arr}

        return sample