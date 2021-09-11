import segmentation_models_pytorch as smp

encoder_weights = 'imagenet'

def get_model(encoder_name):

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=2,
        classes=2,
    )

    # model = smp.FPN(
    #     encoder_name=encoder_name,
    #     encoder_weights=encoder_weights,
    #     in_channels=2,
    #     classes=2,
    # )
    
    return model