import nibabel as nib

# file_path = '../../../INSTANCE/train_2/data/001.nii.gz'
file_path = '../../../INSTANCE/train_2/label/010.nii.gz'

# 读取nii.gz文件
nii_file = nib.load(file_path)

# 获取元数据信息
header = nii_file.header

# 打印元数据信息
print(header)