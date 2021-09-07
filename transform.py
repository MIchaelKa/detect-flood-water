import albumentations as A

def get_train_transform(crop_size):
    transform = A.Compose([

        A.RandomCrop(crop_size, crop_size),

        # A.OpticalDistortion(
        #     distort_limit=(0,1),
        #     shift_limit=0.5,
        #     # border_mode=0,
        #     p=0.5
        # ),

        A.RandomRotate90(),
        A.HorizontalFlip(),
        A.VerticalFlip(),

        # A.RandomSizedCrop(
        #     min_max_height=(256, 512),
        #     height=crop_size,
        #     width=crop_size,
        # )
    ])
    return transform