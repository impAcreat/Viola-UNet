# load data and pre-post precess
import os
from glob import glob
from config import wind_levels, spacing
import config
import numpy as np

from monai.transforms import *
from monai.data import Dataset, DataLoader


pre_process = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        CopyItemsd(keys=["image"], times=2, names=["img_2", "img_3"]),
        ScaleIntensityRanged(
            keys=["image"], a_min=wind_levels[0][0], a_max=wind_levels[0][1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        ScaleIntensityRanged(
            keys=["img_2"], a_min=wind_levels[1][0], a_max=wind_levels[1][1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        ScaleIntensityRanged(
            keys=["img_3"], a_min=wind_levels[2][0], a_max=wind_levels[2][1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        ConcatItemsd(['image', 'img_2', 'img_3'], name='image'),
        DeleteItemsd(['img_2', 'img_3']),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RPI"),  # RPI, RAS
        ToTensord(keys=["image", "label"]),
    ]
)


def load_data():
    images_nii = sorted(glob(os.path.join(config.data_input_folder, "*.nii*")))
    labels_nii = sorted(glob(os.path.join(config.label_input_folder, "*.nii*")))
    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images_nii, labels_nii)]
    
    train_list = data_dicts[:int(len(data_dicts)*config.train_ratio)]
    val_list = data_dicts[int(len(data_dicts)*config.train_ratio):]

    # test_dataset = Dataset(data=test_file_list, transform=pre_process)
    train_dataset = Dataset(data=train_list, transform=pre_process)
    val_dataset = Dataset(data=val_list, transform=pre_process)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    return train_loader, val_loader

## ----------------------------------------------

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, EnsureTyped, Spacingd, Orientationd

# 自定义转换，用于保留 image_meta_dict
class AddMetaDatad:
    def __call__(self, data):
        data["image_meta_dict"] = data["image"].meta
        return data

read_raw_image = Compose(
    [
        LoadImaged(keys=["image"]),
        AddMetaDatad(),  # 添加自定义转换以保留 image_meta_dict
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=100, b_min=0., b_max=1., clip=True),
        EnsureTyped(keys=["image"])
    ]
)

# 测试代码，加载图像并检查元数据
if __name__ == "__main__":
    test_file_list = [{"image": "../../INSTANCE/train_2/data/001.nii.gz"}]
    raw_data = read_raw_image(test_file_list[0])
    raw_img = raw_data["image"]
    pixdims = raw_data["image_meta_dict"]["pixdim"][1:4]
    print(pixdims)
