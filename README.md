## Dir

* docs：
  * analysis of model and code
  * dataset
  * ourwork
  * ......
* Viola-UNet
  * integrated work improved from original code
* mine
  * test
  * pretrain, display, .....



## Usage

* define your file path in ./Viola-Unet/config.py
  * ATTNETION: root file is supposed to be defined correctly
* use main to run



# OURWORK

## detector

### DenseNet

* 网络类似于ResNet，DenseNet将前面所有层与后面层的**密集连接**，通过密集连接（Dense Connectivity）来增强信息流和梯度传播。具体来说，每一层都接收来自所有前面层的特征图，从而实现特征复用和更有效的训练。
* 模型结构：

![image-20240530145719741](./assets/image-20240530145719741.png)

* 功能：分类

![image-20240530150504326](./assets/image-20240530150504326.png)

### Impl

* 在本项目中，densenet作用体现在2个方面：
  * 用于CAM：CAM通过特征图可视化，使CNN更具解释性
  * 对颅内出血进行分类：any_ich, edh, iph, ivh, sah, sdh
  * 在detector检测到出血但Viola-Unet并未预测出时，进行修正
  
  ![image-20240530153412558](./assets/image-20240530153412558.png)
  
* 由于densenet121并未提供预训练参数，我们尝试进行替代

  * torch.densenet121(weights='DEFAULT')：直接使用torch基于ImageNet训练的模型参数
  * 微调：完成了预训练代码，但INSTANCE提供的数据集不包括出血类型信息
  * 改为2分类：是否出现颅内出血
    * 计算label的出血量lesion_volume，得到densenet微调label
    * 相比原模型：丢失了CAM和颅内出血分类功能，但不对segmentation造成影响


> reference
>
> * EigenCam: https://blog.csdn.net/qq_36070656/article/details/131740372



## Viola attn

* 该数据集提供的数据特点：横截面CT清晰度高于其它截面

<img src="./assets/image-20240530161155491.png" alt="image-20240530161155491" style="zoom:80%;" />

* 改进viola-attn计算方式：增加

## Other dataset

* Robustness



