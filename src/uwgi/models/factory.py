from torch import nn

from .unet import UNetSmall


def build_model(
    name: str,
    in_channels: int = 1,
    num_classes: int = 3,
    encoder_weights: str | None = None,
    classification_head: bool = False,
) -> nn.Module:
    if name == "unet_small":
        return UNetSmall(in_channels=in_channels, num_classes=num_classes)

    if name.startswith("smp_"):
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "Install segmentation-models-pytorch and timm to use SMP model names."
            ) from exc

        parts = name.split("_")
        if len(parts) < 3:
            raise ValueError("SMP model name format: smp_<architecture>_<encoder>")
        arch = parts[1]
        encoder = "_".join(parts[2:])
        kwargs = {
            "encoder_name": encoder,
            "encoder_weights": encoder_weights,
            "in_channels": in_channels,
            "classes": num_classes,
        }
        if classification_head:
            kwargs["aux_params"] = {
                "pooling": "avg",
                "dropout": 0.2,
                "activation": None,
                "classes": num_classes,
            }
        if arch == "unet":
            return smp.Unet(**kwargs)
        if arch == "unetplusplus":
            return smp.UnetPlusPlus(**kwargs)
        if arch == "deeplabv3plus":
            return smp.DeepLabV3Plus(**kwargs)
        raise ValueError(f"Unsupported SMP architecture: {arch}")

    raise ValueError(f"Unknown model name: {name}")
