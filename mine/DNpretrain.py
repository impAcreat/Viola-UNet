import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from tqdm import tqdm

from load_data import load_data
import config

def get_densenet121_model(num_classes, device):
    model = models.densenet121(weights='DEFAULT')
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)
    model = model.to(device)
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_model_wts = model.state_dict()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0

            # 迭代数据
            for batch_data in tqdm(dataloader):
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # 反向传播和优化
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f}')

            # 深度拷贝模型
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()

    print('Best val loss: {:4f}'.format(best_loss))

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    ## data
    train_loader, val_loader = load_data()

    ## gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## model
    model = get_densenet121_model(num_classes=2, device=device)

    ## loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    ## 
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device=device)

    torch.save(model.state_dict(), config.dn_model_path)
