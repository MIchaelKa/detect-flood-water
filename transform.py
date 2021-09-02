import albumentations

def get_train_transform(crop_size):
    transform = albumentations.Compose([
        albumentations.RandomCrop(crop_size, crop_size),
        # albumentations.RandomRotate90(),
        # albumentations.HorizontalFlip(),
        # albumentations.VerticalFlip(),
    ])
    return transform