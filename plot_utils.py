import matplotlib.pyplot as plt

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