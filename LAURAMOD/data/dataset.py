import os
import glob
from sklearn.model_selection import train_test_split
from monai.data import Dataset, DataLoader
from .transforms import get_train_transforms, get_val_transforms

def get_dataloaders(config, model_config):
    image_files = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "*.png")))
    mask_files = sorted(glob.glob(os.path.join(config.MASK_DIR, "*.png")))
    data_dicts = [{"image": img, "label": mask} for img, mask in zip(image_files, mask_files)]

    train_files, test_files = train_test_split(data_dicts, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.125, random_state=42)

    train_ds = Dataset(data=train_files, transform=get_train_transforms(model_config["image_size"]))
    val_ds = Dataset(data=val_files, transform=get_val_transforms(model_config["image_size"]))

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_files, val_files, test_files
