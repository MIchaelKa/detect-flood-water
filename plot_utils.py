import matplotlib.pyplot as plt

import numpy as np

#
# Helper functions for visualizing Sentinel-1 images
#
def scale_img(matrix):
    """
    Returns a scaled (H, W, D) image that is visually inspectable.
    Image is linearly scaled between min_ and max_value, by channel.

    Args:
        matrix (np.array): (H, W, D) image to be scaled

    Returns:
        np.array: Image (H, W, 3) ready for visualization
    """
    # Set min/max values
    min_values = np.array([-23, -28, 0.2])
    max_values = np.array([0, -5, 1])

    # min_values = np.array([-28, -23, 1/0.2])
    # max_values = np.array([-5, 0, 1])

    # Reshape matrix
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)

    # Scale by min/max
    matrix = (matrix - min_values[None, :]) / (
        max_values[None, :] - min_values[None, :]
    )
    matrix = np.reshape(matrix, [w, h, d])

    # Limit values to 0/1 interval
    return matrix.clip(0, 1)


def create_false_color_composite(vv_img, vh_img):
    """
    Returns a S1 false color composite for visualization.

    Returns:
        np.array: image (H, W, 3) ready for visualization
    """

    # Stack arrays along the last dimension
    s1_img = np.stack((vv_img, vh_img), axis=-1)

    # Create false color composite
    img = np.zeros((512, 512, 3), dtype=np.float32)
    img[:, :, :2] = s1_img.copy()
    img[:, :, 2] = s1_img[:, :, 0] / s1_img[:, :, 1]

    return scale_img(img)

#
# images
#

import rasterio

def show_image_hist_by_id(chip_id, data):
    row = data[data.chip_id == chip_id].iloc[0]
    show_image_hist(row)

def show_image_hist_by_iloc(idx, data):
    row = data.iloc[idx]
    show_image_hist(row)

def show_image_hist(row):
    print(f'Flood: {row.flood_id}')
    print(f'Chip: {row.chip_id}')

    with rasterio.open(row.vv_path) as vv:
        vv_img = vv.read(1)
    with rasterio.open(row.vh_path) as vh:
        vh_img = vh.read(1)
    with rasterio.open(row.label_path) as lp:
        lp_img = lp.read(1)    
        
    s1_img = create_false_color_composite(vv_img, vh_img)  
    label_to_show = np.ma.masked_where((lp_img == 0) | (lp_img == 255), lp_img)

    _, axes = plt.subplots(1, 3, figsize=(15,5))

    axes[0].set_title("False colors", fontsize=14)
    axes[0].imshow(s1_img)
    axes[0].imshow(label_to_show, cmap="cool", alpha=0.6)
    axes[0].grid(False)
    axes[0].axis('off')

    label_mask = np.ma.masked_where((lp_img == 0) | (lp_img == 255), vv_img)
    ground_mask = np.ma.masked_where((lp_img == 1) | (lp_img == 255), vv_img)
    mask = np.ma.masked_where((lp_img == 1) | (lp_img == 0), vv_img)
    data = [label_mask.compressed(), ground_mask.compressed(), mask.compressed()]
    labels = ['label', 'ground', 'mask']

    axes[1].set_title("VV hist", fontsize=14)
    axes[1].hist(data, bins=50, histtype='barstacked', label=labels)
    axes[1].legend()

    label_mask = np.ma.masked_where((lp_img == 0) | (lp_img == 255), vh_img)
    ground_mask = np.ma.masked_where((lp_img == 1) | (lp_img == 255), vh_img)
    mask = np.ma.masked_where((lp_img == 1) | (lp_img == 0), vh_img)
    data = [label_mask.compressed(), ground_mask.compressed(), mask.compressed()]
    labels = ['label', 'ground', 'mask']

    axes[2].set_title("VH hist", fontsize=14)
    axes[2].hist(data, bins=50, histtype='barstacked', label=labels)
    axes[2].legend()


def get_chip_by_id(chip_id, data):
    row = data[data.chip_id == chip_id].iloc[0]

    with rasterio.open(row.vv_path) as vv:
        vv_img = vv.read(1)
    with rasterio.open(row.vh_path) as vh:
        vh_img = vh.read(1)

    s1_img = create_false_color_composite(vv_img, vh_img)
    # s1_img = create_false_color_composite(vh_img, vv_img)
    return s1_img

def show_chip_by_id(chip_id, data):
    row = data[data.chip_id == chip_id].iloc[0]
    show_data_row(row)

def show_chip_by_index(idx, data):
    row = data.loc[idx]
    show_data_row(row)

def show_chip_by_iloc(idx, data):
    row = data.iloc[idx]
    show_data_row(row)

def show_data_row(row):    
    print(f'Flood: {row.flood_id}')
    print(f'Chip: {row.chip_id}')
    
    with rasterio.open(row.vv_path) as vv:
        vv_img = vv.read(1)
    with rasterio.open(row.vh_path) as vh:
        vh_img = vh.read(1)
    with rasterio.open(row.label_path) as lp:
        lp_img = lp.read(1)    
        

    s1_img = create_false_color_composite(vv_img, vh_img)  
    label_to_show = np.ma.masked_where((lp_img == 0) | (lp_img == 255), lp_img)
    
    label_sum = (1 - label_to_show.mask).sum()
    print(f'Label: {label_sum}')
    
    _, axes = plt.subplots(1, 4, figsize=(20,5))
    
    axes[0].imshow(s1_img)
    axes[0].set_title("False colors", fontsize=14)
    axes[0].grid(False)
    axes[0].axis('off')
    
    axes[1].imshow(vv_img, cmap="gray")
    axes[1].set_title("VV", fontsize=14)
    axes[1].grid(False)
    axes[1].axis('off')
    
    axes[2].imshow(vh_img, cmap="gray")
    axes[2].set_title("VH", fontsize=14)
    axes[2].grid(False)
    axes[2].axis('off')
    
    axes[3].imshow(s1_img)
    axes[3].imshow(label_to_show, cmap="cool", alpha=1)
    axes[3].set_title("Label", fontsize=14)
    axes[3].grid(False)
    axes[3].axis('off')

#
# aux
#

def show_aux_data_by_id(chip_id, data):
    row = data[data.chip_id == chip_id].iloc[0]
    show_aux_data(row)

def show_aux_data_by_iloc(idx, data):
    row = data.iloc[idx]
    show_aux_data(row)

def show_aux_data(row):
    print(f'Flood: {row.flood_id}')
    print(f'Chip: {row.chip_id}')

    features = [
        'vv',
        'vh',
        'nasadem',
        'jrc-gsw-extent',
        'jrc-gsw-occurrence',
        'jrc-gsw-recurrence',
        'jrc-gsw-seasonality',
        'jrc-gsw-transitions',
        'jrc-gsw-change',
    ]

    images = {}

    for feature in features:
        with rasterio.open(row[f'{feature}_path']) as f:
            images[feature] = f.read(1)

    with rasterio.open(row.label_path) as lp:
        lp_img = lp.read(1)

    vv_img, vh_img = images['vv'], images['vh']
    
    s1_img = create_false_color_composite(vv_img, vh_img)  
    label_to_show = np.ma.masked_where((lp_img == 0) | (lp_img == 255), lp_img)
    
    label_sum = (1 - label_to_show.mask).sum()
    print(f'Label: {label_sum}')

    _, axes = plt.subplots(11, 2, figsize=(5*2,5*11))
    
    axes[0,0].imshow(s1_img)
    axes[0,0].set_title("False colors", fontsize=14)
    axes[0,0].grid(False)
    axes[0,0].axis('off')
    
    axes[1,0].imshow(s1_img)
    axes[1,0].imshow(label_to_show, cmap="cool", alpha=1)
    axes[1,0].set_title("Label", fontsize=14)
    axes[1,0].grid(False)
    axes[1,0].axis('off')

    for i, feature in enumerate(features):
        axes[2+i,0].imshow(images[feature], cmap="gray")
        axes[2+i,0].set_title(feature, fontsize=14)
        axes[2+i,0].grid(False)
        axes[2+i,0].axis('off')

    for i, feature in enumerate(features):
        axes[2+i,1].set_title(feature, fontsize=14)
        axes[2+i,1].hist(images[feature].reshape(-1), bins=50)

#
# dataset
#

def show_dataset(dataset, start_index, count, feature_index=0, show_mask=True, show_hist=False):
    
    indices = np.arange(start_index, start_index+count)
    print(indices)

    size = 5
    if show_hist:
        plt.figure(figsize=(count*size,size*2))
        rows = 2
    else:
        plt.figure(figsize=(count*size,size))
        rows = 1
    
    for i, index in enumerate(indices):    

        sample_data = dataset[index]
        chip = sample_data['chip']
        label = sample_data['label']
        label_to_show = np.ma.masked_where((label == 0) | (label == 255), label)

        plt.subplot(rows,count,i+1)
        plt.grid(False)
        plt.axis('off')
        
        label_sum = (1 - label_to_show.mask).sum()
        plt.title(f'Shape: {chip.shape}\nLabel: {label_sum}', fontsize=16)
      
        plt.imshow(chip[feature_index], cmap="gray")
        if show_mask:
            plt.imshow(label_to_show, cmap="cool", alpha=0.5)
        
        if show_hist:
            plt.subplot(rows,count,count+i+1)
            plt.hist(chip[feature_index].reshape(-1), bins=50)

def show_predictions(chip_to_show, chip, label, pred):    
    _, axes = plt.subplots(1, 3, figsize=(15,5))

    fc_index = 1
    label_index = 0
    pred_index = 2
    
    axes[fc_index].set_title("False colors")
    axes[fc_index].grid(False)
    axes[fc_index].axis('off')
    axes[fc_index].imshow(chip_to_show)
    
    axes[label_index].set_title("Label")
    axes[label_index].grid(False)
    axes[label_index].axis('off')
    axes[label_index].imshow(chip[1], cmap='gray')
    axes[label_index].imshow(label, cmap="cool", alpha=1)

    axes[pred_index].set_title("Prediction")
    axes[pred_index].grid(False)
    axes[pred_index].axis('off')
    axes[pred_index].imshow(chip[1], cmap='gray')
    axes[pred_index].imshow(pred, cmap="cool", alpha=1)
    
    # TODO: show intersection and union pixels
    # axes[2].set_title("IoU")

# cmap = gray, bone
def show_image(image, cmap=None):
    # plt.figure(figsize=(5,5))
    plt.grid(False)
    plt.axis('off')
    plt.imshow(image, cmap=cmap)
    plt.show()

def show_image_and_label(image, label, cmap=None):
    # plt.figure(figsize=(5,5))
    plt.grid(False)
    plt.axis('off')
    plt.imshow(image, cmap=cmap)
    plt.imshow(label, cmap="cool", alpha=0.5)
    plt.show()

#
# metrics
#

def show_train_metrics(loss_meter, score_meter):
    _, axes = plt.subplots(1, 2, figsize=(15,5))

    axes[0].set_title("Loss")
    axes[0].plot(loss_meter.history)
    axes[0].plot(loss_meter.moving_average(0.9))

    axes[1].set_title("Scores")
    axes[1].plot(score_meter.history)
    axes[1].plot(score_meter.moving_average(0.9))

    # plt.title('Train metrics')
    plt.plot()

def show_loss_and_score(train_info, start_from=0):
    _, axes = plt.subplots(1, 2, figsize=(15,5))

    valid_iters = train_info['valid_iters'][start_from:]

    axes[0].set_title("Loss")
    axes[0].plot(valid_iters, train_info['train_loss_history'][start_from:], '-o')
    axes[0].plot(valid_iters, train_info['valid_loss_history'][start_from:], '-o')
    # axes[0].set_xticks(valid_iters)
    axes[0].legend(['train', 'val'], loc='upper right')
    
    axes[1].set_title("Scores")
    axes[1].plot(valid_iters, train_info['train_score_history'][start_from:], '-o')
    axes[1].plot(valid_iters, train_info['valid_score_history'][start_from:], '-o')
    axes[1].legend(['train', 'val'], loc='lower right')

def show_valid_score_by_flood(train_info):  
    valid_iters = train_info['valid_iters']
    valid_score_by_flood_id = train_info['valid_score_by_flood_id']
    flood_ids = list(valid_score_by_flood_id.keys())

    _, axes = plt.subplots(1, len(flood_ids), figsize=(20,5))

    for i, flood_id in enumerate(flood_ids):
        axes[i].set_title(flood_id)
        axes[i].plot(valid_iters, valid_score_by_flood_id[flood_id], '-o')


#
# other
#

def create_fcc_dataset_sample(s1_img):
    s1_img = np.transpose(s1_img, [1, 2, 0])
    
    img = np.zeros((512, 512, 3), dtype=np.float32)
    img[:, :, :2] = s1_img.copy()
    
    b_channel = s1_img[:, :, 0] / s1_img[:, :, 1]
    bc_max = b_channel.max()
    bc_min = b_channel.min()
    b_channel = (b_channel - bc_min) / (bc_max - bc_min)       
    img[:, :, 2] = b_channel

    return img