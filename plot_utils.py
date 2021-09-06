import matplotlib.pyplot as plt

import numpy as np 

def show_dataset(dataset, start_index, count):
    
    indices = np.arange(start_index, start_index+count)
    print(indices)
    
    size = 5
    plt.figure(figsize=(count*size,size))
    
    for i, index in enumerate(indices):    

        sample_data = dataset[index]
        chip = sample_data['chip']
        label = sample_data['label']
        label_to_show = np.ma.masked_where((label == 0) | (label == 255), label)

        plt.subplot(1,count,i+1)
        plt.grid(False)
        plt.axis('off')
        
        label_sum = (1 - label_to_show.mask).sum()
        plt.title(f'Shape: {chip.shape}\nLabel: {label_sum}', fontsize=16)
      
        plt.imshow(chip[1], cmap="gray")
        plt.imshow(label_to_show, cmap="cool", alpha=0.5)

def show_predictions(chip, label, pred):    
    _, axes = plt.subplots(1, 2, figsize=(10,5))
    
    axes[0].set_title("Label")
    axes[0].grid(False)
    axes[0].axis('off')
    axes[0].imshow(chip[1], cmap='gray')
    axes[0].imshow(label, cmap="cool", alpha=1)

    axes[1].set_title("Prediction")
    axes[1].grid(False)
    axes[1].axis('off')
    axes[1].imshow(chip[1], cmap='gray')
    axes[1].imshow(pred, cmap="cool", alpha=0.5)
    
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