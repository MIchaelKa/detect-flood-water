import matplotlib.pyplot as plt

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

def show_loss_and_score(train_info):
    _, axes = plt.subplots(1, 2, figsize=(15,5))

    valid_iters = train_info['valid_iters']

    axes[0].set_title("Loss")
    axes[0].plot(valid_iters, train_info['train_loss_history'], '-o')
    axes[0].plot(valid_iters, train_info['valid_loss_history'], '-o')
    # axes[0].set_xticks(valid_iters)
    axes[0].legend(['train', 'val'], loc='upper right')
    
    axes[1].set_title("Scores")
    axes[1].plot(valid_iters, train_info['train_score_history'], '-o')
    axes[1].plot(valid_iters, train_info['valid_score_history'], '-o')
    axes[1].legend(['train', 'val'], loc='lower right')