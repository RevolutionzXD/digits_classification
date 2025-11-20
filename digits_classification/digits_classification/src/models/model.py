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