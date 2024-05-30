import torch
import torch.nn as nn
from torchvision import models

model = models.densenet121(pretrained=True)

# change the classifier（to 10, for example）
# num_features = model.classifier.in_features
# model.classifier = nn.Linear(num_features, 10)  # 10 是新的类别数

# model structure
print(model)

# use gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# random input
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# test 
output = model(dummy_input)
print(output)