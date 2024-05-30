## DenseNet121





## Appendix

### state_dict()

* 预训练权重文件中的状态字典 (`state_dict`) 是一个包含了模型所有参数（权重和偏置）的Python字典。这个字典的键是模型各层的名称，值是这些层对应的参数张量

* PyTorch中用于保存和加载模型的参数

* code

  * save

  ```python
  torch.save(model.state_dict(), 'saved_path.pth')
  ```

  * load

  ```python
  model.load_state_dict(torch.load('saved_path.pth'))
  ```

* state_dict的作用

  - **模块化**：通过 `state_dict` 可以非常方便地保存和加载模型的参数，而不需要保存整个模型。这使得模型的存储和加载更加灵活和高效。
  - **迁移学习**：使用预训练的 `state_dict` 可以快速加载预训练的权重，进行迁移学习。这使得在新的任务中使用预训练模型变得非常方便。
  - **断点续训**：在训练过程中，可以定期保存 `state_dict`，如果训练被中断，可以从保存的 `state_dict` 恢复训练，而不需要从头开始。

> reference:
>
> * pytorch source: https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html#torchvision.models.DenseNet121_Weights