import numpy as np

class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.total_sum = 0
        self.total_count = 0

    def update(self, x):
        self.history.append(x)
        self.total_sum += x
        self.total_count += 1

    def compute_average(self):
        return np.mean(self.history)

    def moving_average(self, alpha):
        avg_history = [self.history[0]]
        for i in range(1, len(self.history)):
            moving_avg = alpha * avg_history[-1] + (1 - alpha) * self.history[i]
            avg_history.append(moving_avg)
        return avg_history


import torch

# TODO: does converting to cpu make computation longer?
def intersection_and_union(pred, true):
    """
    Calculates intersection and union for a batch of images.

    Args:
        pred (torch.Tensor): a tensor of predictions
        true (torc.Tensor): a tensor of labels

    Returns:
        intersection (int): total intersection of pixels
        union (int): total union of pixels
    """
    valid_pixel_mask = true.ne(255)  # valid pixel mask
    true = true.masked_select(valid_pixel_mask).to("cpu")
    pred = pred.masked_select(valid_pixel_mask).to("cpu")

    # Intersection and union totals
    intersection = np.logical_and(true, pred)
    # correct_pixels = (pred == true).sum()
    # print(intersection.sum(), correct_pixels)
    union = np.logical_or(true, pred)
    return intersection.sum(), union.sum()

class IoUMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []

        self.intersection = 0
        self.union = 0

    def compute_prediction(self, output):
        preds = torch.softmax(output, dim=1)[:, 1]
        preds = (preds > 0.5) * 1
        return preds

    def update(self, output, target):
        preds = self.compute_prediction(output)
        intersection, union = intersection_and_union(preds, target)

        self.intersection += intersection
        self.union += union

        # TODO: add to history?
        batch_iou = intersection / union
        # print(batch_iou)

    def compute_score(self):
        return self.intersection / self.union

        

