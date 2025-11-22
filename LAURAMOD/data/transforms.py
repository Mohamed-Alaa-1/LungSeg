import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    RandAffined,
    RandRotate90d,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    EnsureTyped,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandFlipd,
)

def get_train_transforms(image_size):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(image_size, image_size)),
        RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.3),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.3),
        RandAffined(
            keys=['image', 'label'], prob=0.3, translate_range=(5, 5),
            rotate_range=(np.pi / 36, np.pi / 36), scale_range=(0.05, 0.05),
            mode=('bilinear', 'nearest'),
        ),
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.005),
        RandScaleIntensityd(keys=["image"], factors=0.05, prob=0.2),
        EnsureTyped(keys=["image", "label"], track_meta=False),
    ])

def get_val_transforms(image_size):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(image_size, image_size)),
        EnsureTyped(keys=["image", "label"], track_meta=False),
    ])
