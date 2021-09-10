import albumentations as A

def get_train_transform(crop_size):
    transform = A.Compose([

        A.RandomCrop(crop_size, crop_size),

        A.RandomBrightness(limit=0.3, p=0.5)
        # A.RandomBrightness(limit=(1,1), p=0.5)

        # A.OpticalDistortion(
        #     distort_limit=(0,1),
        #     shift_limit=0.5,
        #     # border_mode=0,
        #     p=0.5
        # ),

        # A.RandomRotate90(),
        # A.HorizontalFlip(),
        # A.VerticalFlip(),

        # A.RandomSizedCrop(
        #     min_max_height=(256, 512),
        #     height=crop_size,
        #     width=crop_size,
        # )
        ],
        additional_targets={'invalid_mask': 'mask'}
    )
    return transform