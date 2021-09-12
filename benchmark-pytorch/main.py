import os
from pathlib import Path
from tqdm import tqdm
from tifffile import imwrite

import torch

import numpy as np
import segmentation_models_pytorch as smp
import rasterio

ROOT_DIRECTORY = Path("/codeexecution")
# ROOT_DIRECTORY = Path("./")


SUBMISSION_DIRECTORY = ROOT_DIRECTORY / "submission"
ASSETS_DIRECTORY = ROOT_DIRECTORY / "assets"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
INPUT_IMAGES_DIRECTORY = DATA_DIRECTORY / "test_features"

# Make sure the smp loader can find our torch assets because we don't have internet!
os.environ["TORCH_HOME"] = str(ASSETS_DIRECTORY / "torch")

def get_model():

    encoder_name = 'timm-efficientnet-b0'

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=2,
        classes=2,
    )

    model.load_state_dict(torch.load(ASSETS_DIRECTORY / "unet_timm-efficientnet-b0_11.09_fold_2.pth"))
    # model.load_state_dict(torch.load(ASSETS_DIRECTORY / "unet_timm-efficientnet-b0_11.09_fold_2.pth", map_location='cpu'))
    model.eval()

    return model

def predict(model, vv_path, vh_path):
        torch.set_grad_enabled(False)

        # Create a 2-channel image
        with rasterio.open(vv_path) as vv:
            vv_img = vv.read(1)
        with rasterio.open(vh_path) as vh:
            vh_img = vh.read(1)
        x_arr = np.stack([vv_img, vh_img], axis=-1)

        # Min-max normalization
        min_norm = -77
        max_norm = 26
        x_arr = np.clip(x_arr, min_norm, max_norm)
        x_arr = (x_arr - min_norm) / (max_norm - min_norm)

        # Transpose
        x_arr = np.transpose(x_arr, [2, 0, 1])
        x_arr = np.expand_dims(x_arr, axis=0)

        # Perform inference
        output = model(torch.from_numpy(x_arr))
        preds = compute_prediction(output)
        return preds.detach().numpy().squeeze()#.squeeze()

def compute_prediction(output):
    preds = torch.softmax(output, dim=1)[:, 1]
    preds = (preds > 0.5) * 1
    return preds

def get_expected_chip_ids():
    """
    Use the test features directory to see which images are expected.
    """
    paths = INPUT_IMAGES_DIRECTORY.glob("*.tif")
    # Return one chip id per two bands (VV/VH)
    ids = list(sorted(set(path.stem.split("_")[0] for path in paths)))
    return ids

def make_prediction(chip_id, model):
    """
    Given a chip_id, read in the vv/vh bands and predict a water mask.

    Args:
        chip_id (str): test chip id

    Returns:
        output_prediction (arr): prediction as a numpy array
    """
    # print("Starting inference.")
    try:
        vv_path = INPUT_IMAGES_DIRECTORY / f"{chip_id}_vv.tif"
        vh_path = INPUT_IMAGES_DIRECTORY / f"{chip_id}_vh.tif"
        output_prediction = predict(model, vv_path, vh_path)
        # print(output_prediction.shape)
    except Exception as e:
        print(f"No bands found for {chip_id}. {e}")
        raise
    return output_prediction

def main():
    print("Loading model")
    # Explicitly set where we expect smp to load the saved resnet from just to be sure
    torch.hub.set_dir(ASSETS_DIRECTORY / "torch/hub")
    model = get_model()

    print("Finding chip IDs")
    chip_ids = get_expected_chip_ids()
    if not chip_ids:
        print("No input images found!")
        return
    
    print(f"Found {len(chip_ids)} test chip_ids. Generating predictions.")
    for chip_id in tqdm(chip_ids, miniters=25):
        output_path = SUBMISSION_DIRECTORY / f"{chip_id}.tif"
        output_data = make_prediction(chip_id, model).astype(np.uint8)
        imwrite(output_path, output_data, dtype=np.uint8)


if __name__ == "__main__":
    main()