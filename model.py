import segmentation_models_pytorch as smp
import ttach as tta

def get_model(encoder_name):

    encoder_weights = 'imagenet' # None, imagenet

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=2,
        classes=2,
    )

    # model = smp.DeepLabV3(
    #     encoder_name=encoder_name,
    #     encoder_weights=encoder_weights,
    #     in_channels=2,
    #     classes=2,
    # )

    # decoder_channels = 256
    # model.decoder = smp.deeplabv3.decoder.DeepLabV3Decoder(
    #     in_channels=model.encoder.out_channels[-1],
    #     out_channels=decoder_channels,
    #     atrous_rates=(6, 12, 18)
    # )

    # model = smp.FPN(
    #     encoder_name=encoder_name,
    #     encoder_weights=encoder_weights,
    #     in_channels=2,
    #     classes=2,
    # )
    
    return model

#
# TTA
#

def channel_shuffle(img):
    img = img[:,(1,0),...]
    return img

class TTAChannelShuffle(tta.base.ImageOnlyTransform):
    """Rearrange channels of the input image"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
#         print(f'shape 1: {image.shape}')
        if apply:
            image = channel_shuffle(image)
#         print(f'shape 2: {image.shape}')
        return image

def get_model_tta(model):
    # transforms = tta.Compose([
    #         # tta.Scale(scales=[1, 2]),
    #         # tta.Multiply(factors=[0.9, 1.1]),
    #         # TTAChannelShuffle()
    #     ]
    # )

    transforms = tta.aliases.d4_transform()
    # transforms = tta.aliases.hflip_transform()

    tta_model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')

    return tta_model