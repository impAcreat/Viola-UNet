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

### Result

* 由于model预测lesion_volume<0的情况非常少，故该部分对整体影响微乎其微


> reference
>
> * EigenCam: https://blog.csdn.net/qq_36070656/article/details/131740372



## Viola attn

* 该数据集提供的数据特点：横截面CT清晰度高于其它截面

<img src="./assets/image-20240530161155491.png" alt="image-20240530161155491" style="zoom:80%;" />

### Viola-Attn

* 改进**viola-attn**计算方式

  * **Origin Viola_attn**：

  ```python
          viola_j = xs * ys + ys * zs + zs * xs       # 0-3
          viola_m = xs * ys * zs                  # 0-1  
          viola_a = self.relu(xt + yt + zt)       # 0-3
  
          viola = viola_j + viola_m + viola_a
          viola = 0.1 * viola + 0.3 
          viola = viola + l2norm(viola.contiguous().view(b,-1)).view(b,c,h,w,d)  
  ```

  * **New:**增加横截面的权重 & 尝试优化的超参数
    * 在增加权重后，涉及到归一化的问题（由于原来通过sigmoid后xs、ys、zs在[0, 1]区间，相乘后仍符合），尝试2种解决方式：
      * 激活函数再处理：sigmoid，tanh（relu）
      * 忽略

  ```python
  # weighted 
  viola_j = xy_weight * xs * ys + yz_weight * ys * zs + xz_weight * zs * xs
  
  # viola_j = activate_func(viola_j)
          
  # alter hyperparameters
  viola = viola_k * viola + viola_b
  ```

### fine-tuning

* 在更改Viola-Attn后，需要对模型微调（或者重新训练）
  * current：仅对截面加权，并未修改超参

### Result

* 仅进行了一次完整的预训练+预测，params：

  > xy_weight: 0.4*2.5=1
  >
  > yz_weight: 0.3*2.5=7.5
  >
  > xz_weight: 0.3*2.5=7.5
  >
  > activate_again: false
  >
  > viola_k: 0.1
  >
  > viola_b: 0.3

* 效果较之前差别不大：DSC略高，HD几乎相同（RVD，NSD暂未考虑）

## Other dataset

* Robustness



