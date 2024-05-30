## NIfTI

* NIfTI（Neuroimaging Informatics Technology Initiative）是目前各大神经影像分析工具普遍兼容的**体素水平**的数据格式，用于存储和分享医学扫描数据，如MRI、CT扫描等。NIfTI格式支持三维（3D）或四维（4D）的数据存储，四维数据通常用于包含时间序列的扫描，比如功能性MRI（fMRI）
* 在医学影像处理中，将NIfTI文件压缩为`.nii.gz`格式是常见的做法，因为它可以显著减少磁盘空间的占用，同时保留图像数据和元数据的完整性

## Backup Dataset: DeepLesion

* 32120 CT slices of 4427 patients
* 包含病变注释的关键CT切片 & 3D背景(关键切片上下各30mm的额外切片)
* 与 INSTANCE 2022不同，DeepLesion包含大量CT数据，但分布于身体各个部位
  * 故：如果使用DeepLesion，可能将模型目的调整为DeepLesion数据集包含的某身体部位检测，并对模型做出针对性调整