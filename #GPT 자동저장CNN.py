# binary_cnn_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import classification_report
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(
        root="C:/Users/최지완/Desktop/과제연구/과제연구1-1 사진/10개 분류/train",
        transform=transform)
    test_dataset = datasets.ImageFolder(
        root="C:/Users/최지완/Desktop/과제연구/과제연구1-1 사진/10개 분류/test",
        transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.relu = nn.ReLU()
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model_path = "C:/Users/최지완/Desktop/asdfsaFD/model.pth"
    model = MyModel().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("모델을 성공적으로 불러왔습니다.")
    else:
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        num_epochs = 1

        for epoch in range(num_epochs):
            for batch_idx, (images, labels) in enumerate(train_loader, 1):
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                print(f"epoch: {epoch} 진행률:{batch_idx}/{len(train_loader)}", end='\r')

        torch.save(model.state_dict(), model_path)
        print("\n모델을 저장했습니다.")

    model.eval()
    with torch.no_grad():

        all_labels = []
        pred_labels = []

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.35 

            all_labels.append(labels.detach().cpu().numpy())
            pred_labels.append(preds.detach().cpu().numpy())
        
        labels_np = np.concatenate(all_labels)
        preds_np = np.concatenate(pred_labels)

        print(classification_report(labels_np, preds_np, target_names=['class0', 'class1']))
        print(np.unique(labels_np, return_counts=True))  # 실제 라벨 분포
        print(np.unique(preds_np, return_counts=True))   # 예측 분포

if __name__ == "__main__":
    main()
