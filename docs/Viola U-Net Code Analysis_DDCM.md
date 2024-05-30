# Viola U-Net Code Analysis: DDCM

![image-20240507200111004](./assets/image-20240507200111004.png)

## DCs

### Code

```python
self.features.append(nn.Sequential(
                nn.Conv2d(self.in_dim + idx * out_dim,
                            out_dim,
                            kernel_size=(kernel, 1),
                            dilation=rate,
                            stride=(self.strides[idx],1),
                            padding=(rate * (kernel - 1) // 2, 0),
                            bias=bias),
                nn.Dropout(p=dropout)
                )
                )
```

* 存储：**self.features**
* 结构：nn.Sequential
  * nn.Conv2d
  * nn.Dropout

### Conv2d

<div class="table-box"><table><thead><tr><th align="left">参数</th><th align="left">参数类型</th><th align="left"></th><th align="left"></th></tr></thead><tbody><tr><td align="left"><code>in_channels</code></td><td align="left">int</td><td align="left">Number of channels in the input image</td><td align="left">输入图像通道数</td></tr><tr><td align="left"><code>out_channels</code></td><td align="left">int</td><td align="left">Number of channels produced by the convolution</td><td align="left">卷积产生的通道数</td></tr><tr><td align="left"><code>kernel_size</code></td><td align="left">(int or tuple)</td><td align="left">Size of the convolving kernel</td><td align="left">卷积核尺寸，可以设为1个int型数或者一个(int, int)型的元组。例如(2,3)是高2宽3卷积核</td></tr><tr><td align="left"><code>stride</code></td><td align="left">(int or tuple, optional)</td><td align="left">Stride of the convolution. Default: 1</td><td align="left">卷积步长，默认为1。可以设为1个int型数或者一个(int, int)型的元组。</td></tr><tr><td align="left"><code>padding</code></td><td align="left">(int or tuple, optional)</td><td align="left">Zero-padding added to both sides of the input. Default: 0</td><td align="left">填充操作，控制<code>padding_mode</code>的数目。</td></tr><tr><td align="left"><code>padding_mode</code></td><td align="left">(string, optional)</td><td align="left">‘zeros’, ‘reflect’, ‘replicate’ or ‘circular’. Default: ‘zeros’</td><td align="left"><code>padding</code>模式，默认为Zero-padding 。</td></tr><tr><td align="left"><code>dilation</code></td><td align="left">(int or tuple, optional)</td><td align="left">Spacing between kernel elements. Default: 1</td><td align="left">扩张操作：控制kernel点（卷积核点）的间距，默认值:1。</td></tr><tr><td align="left"><code>groups</code></td><td align="left">(int, optional)</td><td align="left">Number of blocked connections from input channels to output channels. Default: 1</td><td align="left">group参数的作用是控制分组卷积，默认不分组，为1组。</td></tr><tr><td align="left"><code>bias</code></td><td align="left">(bool, optional)</td><td align="left">If True, adds a learnable bias to the output. Default: True</td><td align="left">为真，则在输出中添加一个可学习的偏差。默认：True。</td></tr></tbody></table></div>

* **in_channel = self.in_dim + idx * out_dim**

> * `self.in_dim`是输入数据的通道数
>
> * `out_dim`是每个卷积层的输出通道数
>
> * `idx`是当前卷积层的索引（从0开始）
>
>   设计原因：使用深度可分离卷积方式（可见convolution总结），在深度可分离卷积中，每个输入通道都被单独地卷积处理，然后结果被叠加在一起。这种操作可以减少计算量和模型参数，同时保持良好的模型性能。此处，每个卷积层的输入通道数逐渐增加，因为每个卷积层的输出都被叠加到下一个卷积层的输入中。

* **kernel = (kernel, 1) = (3, 1)**

> * 在viola将数据输入ddcm之前，已使用AdaAvgPooling将信息压缩到3个维度
> * 卷积核宽度设置为1：
>   * 仅在1个方向上进行卷积操作
>   * 这种设计通常用于处理序列数据，如时间序列或文本数据，因为这些数据通常只在一个维度上有序（如时间或文本的顺序）
> * 故该kernel使conv仅提取对应维度的信息

* **padding=(rate * (kernel - 1) // 2, 0)**：same conv



## Merging

### Code

```
self.conv1x1_out = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(self.in_dim*2 + out_dim * self.num, self.in_dim,  kernel_size=1, bias=False),
        )
```

* **nn.SiLU**：sigmoid层
* **nn.Conv2d**：压缩多通道信息
  * 将输入的多通道信息压缩到少量通道中，可以看到kernel为（1，1）
  * 故：各个DC堆叠的信息通过多通道存储，再通过卷积实现信息的merging



## forward

* squeeze & unsqueeze：使维度符合要求，最终得到dimension = [batch_size, channel, height, width]

### Code

```
xc = x.clone()

for f in self.features:
	x = torch.cat([F.interpolate(f(x), (H, W), mode='bilinear', align_corners=False), x], 1)

x = self.conv1x1_out(torch.cat([xc, x], 1))
```

> `F.interpolate(f(x), (H, W), mode='bilinear', align_corners=False)`表示对`f(x)`的结果进行双线性插值，使其大小变为`(H, W)`。这是为了确保所有特征图的大小都是一致的，以便于在通道维度上进行连接。interpolate：插值操作，在图像领域用于改变图像的大小

> reference:
>
> * 插值：[一篇文章为你讲透双线性插值 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/110754637)