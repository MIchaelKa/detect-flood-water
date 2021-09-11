import numpy as np

class BaseMeter():

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []

    def moving_average(self, alpha):
        avg_history = [self.history[0]]
        for i in range(1, len(self.history)):
            moving_avg = alpha * avg_history[-1] + (1 - alpha) * self.history[i]
            avg_history.append(moving_avg)
        return avg_history


class AverageMeter(BaseMeter):

    def reset(self):
        super().reset()
        self.total_sum = 0
        self.total_count = 0

    def update(self, x):
        self.history.append(x)
        self.total_sum += x
        self.total_count += 1

    def compute_average(self):
        return np.mean(self.history)


# import torch
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

class IoUMeter(BaseMeter):

    def reset(self):
        super().reset()
        self.intersection = 0
        self.union = 0

        self.intersection_by_flood_id = {}
        self.union_by_flood_id = {}

    def update(self, preds, target):
        intersection, union = intersection_and_union(preds, target)

        self.intersection += intersection
        self.union += union

        batch_iou = intersection / union
        self.history.append(batch_iou)
        return batch_iou, intersection, union

    def update_with_flood_id(self, preds, target, flood_id_batch):
        intersection, union = intersection_and_union(preds, target)

        self.intersection += intersection
        self.union += union

        batch_iou = intersection / union
        self.history.append(batch_iou)

        # print(preds.shape, target.shape)

        # IoU by floods
        flood_ids = np.unique(flood_id_batch)
        for flood_id in flood_ids:
            if flood_id not in self.intersection_by_flood_id:
                # print(f'new flood_id: {flood_id}')
                self.intersection_by_flood_id[flood_id] = 0
                self.union_by_flood_id[flood_id] = 0

            batch_mask = (flood_id_batch == flood_id)
            intersection, union = intersection_and_union(preds[batch_mask], target[batch_mask])

            self.intersection_by_flood_id[flood_id] += intersection
            self.union_by_flood_id[flood_id] += union
  
        return batch_iou

    def compute_score(self):
        return self.intersection / self.union

    def compute_score_by_flood_id(self):
        scores = {}
        for key in self.intersection_by_flood_id.keys():
            iou = self.intersection_by_flood_id[key] / self.union_by_flood_id[key]
            scores[key] = iou.item()
        return scores

