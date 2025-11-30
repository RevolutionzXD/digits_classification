import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(SimpleMLP, self).__init__()
        # Lớp 1: Tuyến tính từ 784 điểm ảnh -> 128 đặc điểm
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Lớp 2: Tuyến tính từ 128 đặc điểm -> 10 kết quả (số 0-9)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 1. Làm phẳng ảnh (Flatten): Biến ảnh [Batch, 1, 28, 28] thành [Batch, 784]
        x = torch.flatten(x, 1)
        
        # 2. Qua lớp ẩn thứ nhất + Hàm kích hoạt ReLU
        x = self.fc1(x)
        x = F.relu(x)
        
        # 3. Qua lớp đầu ra
        x = self.fc2(x)
        
        return x
    
class CNN(nn.Module):
    def __init__(self, output_dim=10):
        super(CNN, self).__init__()
        self.first_convert = nn.Conv2d(1, 32, 3, padding = 1)
        self.first_batch = nn.BatchNorm2d(32)

        self.second_convert = nn.Conv2d(32, 64, 3, padding = 1)
        self.second_batch = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.first_fully_connect = nn.Linear(64 * 7 * 7, 128)
        self.second_fully_connect = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.pool(torch.relu(self.first_batch(self.first_convert(x))))
        x = self.pool(torch.relu(self.second_batch(self.second_convert(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.first_fully_connect(x)))
        return self.second_fully_connect(x)