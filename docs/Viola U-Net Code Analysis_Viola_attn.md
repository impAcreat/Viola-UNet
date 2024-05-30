# Viola U-Net Code Analysis:  Viola attn

## Init

（easy to understand）

## Viola attn

![image-20240509143839227](./assets/image-20240509143839227.png)

* **view -> xs, ys, zs, xt, yt, zt**：统一矩阵维度，方便后续处理

* **Viola_attn**：

```python
		viola_j = xs * ys + ys*zs + zs*xs       # 0-3
        viola_m = xs * ys * zs                  # 0-1  
        viola_a = self.relu(xt + yt + zt)       # 0-3

        viola = viola_j + viola_m + viola_a
        viola = 0.1 * viola + 0.3 
        viola = viola + l2norm(viola.contiguous().view(b,-1)).view(b,c,h,w,d)  
```

* 最后返回：**x * viola**



## Possible Improvement

### Viola attn 机制

* 可以看到当前的 viola attn 机制是相对死板的，我们考虑是否可以通过增加权重，修改运算方式的方法改进viola attn。使其更关注某个表征维度，或者利用Transformer attn的思路使模型关注重点部分
* 超参数的调整：在代码中可以看到非常多的超参数被固定了，如：卷积核、卷积步长、空洞卷积的膨胀率，我们想尝试通过调参的方式使模型更适合当前任务

### 模型的结构

* 之前提过viola模型的一个优势就是灵活可配置，可以调整：

  * U-Net 结构
  * 是否对称
  * .......

  

