import segmentation_models_pytorch as smp

encoder_weights = 'imagenet'

def get_model(encoder_name):

    # model = smp.Unet(
    #     encoder_name=encoder_name,
    #     encoder_weights=encoder_weights,
    #     in_channels=2,
    #     classes=2,
    # )

    model = smp.DeepLabV3(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=2,
        classes=2,
    )

    decoder_channels = 256
    model.decoder = smp.deeplabv3.decoder.DeepLabV3Decoder(
        in_channels=model.encoder.out_channels[-1],
        out_channels=decoder_channels,
        atrous_rates=(6, 12, 18)
    )

    # model = smp.FPN(
    #     encoder_name=encoder_name,
    #     encoder_weights=encoder_weights,
    #     in_channels=2,
    #     classes=2,
    # )
    
    return model