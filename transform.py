import albumentations as A

class InvertScaledImg(A.ImageOnlyTransform):
    def apply(self, img, **params):
        return 1 - img

    def get_transform_init_args_names(self):
        return ()

def get_train_transform(crop_size):
    transform = A.Compose([

        A.RandomCrop(crop_size, crop_size),

        # A.HueSaturationValue(
        #     hue_shift_limit=0,
        #     sat_shift_limit=0.3,
        #     val_shift_limit=0.3,
        #     p=0.5
        # )

        # A.RandomRotate90(),
        # A.HorizontalFlip(),
        # A.VerticalFlip(),

        # A.ChannelShuffle(),

        # A.ColorJitter(),

        # A.RGBShift(
        #     r_shift_limit=0.4,
        #     g_shift_limit=0.4,
        #     b_shift_limit=0.4,
        #     p=0.5
        # ),

        # InvertScaledImg(p=0.2),

        # A.RandomBrightness(limit=0.3, p=0.5),
        # A.RandomBrightness(limit=(1,1), p=0.5)

        # A.RandomBrightnessContrast(
        #     brightness_limit=0.2,
        #     contrast_limit=0.2,
        #     p=0.5
        # ),

        # A.OpticalDistortion(
        #     distort_limit=(0,1),
        #     shift_limit=0.5,
        #     # border_mode=0,
        #     p=0.5
        # ),

        # A.RandomSizedCrop(
        #     min_max_height=(256, 512),
        #     height=crop_size,
        #     width=crop_size,
        # )
        ],
        # additional_targets={'invalid_mask': 'mask'}
    )
    return transform