import matplotlib.pyplot as plt
from monai.transforms import LoadImage

# 创建一个 LoadImage 转换器
loader = LoadImage()

# 使用转换器读取图像
image = loader('predict/001.nii.gz')
print(image.shape)

# 使用 matplotlib 显示图像的第一个切片
plt.imshow(image[0], cmap='gray')
plt.show()